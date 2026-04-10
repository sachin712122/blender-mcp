"""
claude_client.py – Interactive Blender-MCP client powered by the Claude API.

Usage
-----
Set the following environment variables before running:

    ANTHROPIC_API_KEY   – Your Anthropic / Foundry API key
    ANTHROPIC_BASE_URL  – (optional) Custom base URL for Azure / Foundry endpoints
    ANTHROPIC_MODEL     – (optional) Model deployment name (default: claude-3-5-sonnet-20241022)
    BLENDER_HOST        – (optional) Blender addon host (default: localhost)
    BLENDER_PORT        – (optional) Blender addon port (default: 9876)

Then run:

    python claude_client.py
"""

import asyncio
import base64
import json
import logging
import os
import socket
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict
from urllib.parse import urlparse

# ---------------------------------------------------------------------------
# Try AnthropicFoundry first (Azure / custom endpoints), fall back to standard
# Anthropic client so the file works with both SDKs.
# ---------------------------------------------------------------------------
try:
    from anthropic import AnthropicFoundry as _Client  # type: ignore
except ImportError:
    from anthropic import Anthropic as _Client  # type: ignore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("BlenderClaudeClient")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEFAULT_HOST = "localhost"
DEFAULT_PORT = 9876
DEFAULT_MODEL = "claude-3-5-sonnet-20241022"


# ---------------------------------------------------------------------------
# BlenderConnection (copied from server.py so this file is self-contained)
# ---------------------------------------------------------------------------

@dataclass
class BlenderConnection:
    host: str
    port: int
    sock: socket.socket = None

    def connect(self) -> bool:
        if self.sock:
            return True
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.connect((self.host, self.port))
            logger.info(f"Connected to Blender at {self.host}:{self.port}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Blender: {e}")
            self.sock = None
            return False

    def disconnect(self):
        if self.sock:
            try:
                self.sock.close()
            except Exception as e:
                logger.error(f"Error disconnecting from Blender: {e}")
            finally:
                self.sock = None

    def _receive_full_response(self, buffer_size: int = 8192) -> bytes:
        chunks = []
        self.sock.settimeout(180.0)
        try:
            while True:
                try:
                    chunk = self.sock.recv(buffer_size)
                    if not chunk:
                        if not chunks:
                            raise Exception("Connection closed before receiving any data")
                        break
                    chunks.append(chunk)
                    try:
                        data = b"".join(chunks)
                        json.loads(data.decode("utf-8"))
                        return data
                    except json.JSONDecodeError:
                        continue
                except socket.timeout:
                    break
                except (ConnectionError, BrokenPipeError, ConnectionResetError) as e:
                    logger.error(f"Socket connection error during receive: {e}")
                    raise
        except socket.timeout:
            pass
        except Exception as e:
            logger.error(f"Error during receive: {e}")
            raise

        if chunks:
            data = b"".join(chunks)
            try:
                json.loads(data.decode("utf-8"))
                return data
            except json.JSONDecodeError:
                raise Exception("Incomplete JSON response received")
        raise Exception("No data received")

    def send_command(self, command_type: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        if not self.sock and not self.connect():
            raise ConnectionError("Not connected to Blender")

        command = {"type": command_type, "params": params or {}}
        try:
            self.sock.sendall(json.dumps(command).encode("utf-8"))
            response_data = self._receive_full_response()
            response = json.loads(response_data.decode("utf-8"))
            if response.get("status") == "error":
                raise Exception(response.get("message", "Unknown error from Blender"))
            return response.get("result", {})
        except socket.timeout:
            self.sock = None
            raise Exception("Timeout waiting for Blender response")
        except (ConnectionError, BrokenPipeError, ConnectionResetError) as e:
            self.sock = None
            raise Exception(f"Connection to Blender lost: {e}")
        except json.JSONDecodeError as e:
            self.sock = None
            raise Exception(f"Invalid response from Blender: {e}")
        except Exception as e:
            self.sock = None
            raise Exception(f"Communication error with Blender: {e}")


# ---------------------------------------------------------------------------
# Global connection helper
# ---------------------------------------------------------------------------
_blender_connection: BlenderConnection = None


def get_blender_connection() -> BlenderConnection:
    global _blender_connection
    if _blender_connection is not None:
        try:
            _blender_connection.send_command("get_polyhaven_status")
            return _blender_connection
        except Exception:
            try:
                _blender_connection.disconnect()
            except Exception:
                pass
            _blender_connection = None

    host = os.getenv("BLENDER_HOST", DEFAULT_HOST)
    port = int(os.getenv("BLENDER_PORT", DEFAULT_PORT))
    _blender_connection = BlenderConnection(host=host, port=port)
    if not _blender_connection.connect():
        _blender_connection = None
        raise Exception("Could not connect to Blender. Make sure the Blender addon is running.")
    return _blender_connection


# ---------------------------------------------------------------------------
# Tool definitions for the Claude API
# ---------------------------------------------------------------------------
TOOLS = [
    {
        "name": "get_scene_info",
        "description": "Get detailed information about the current Blender scene.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "get_object_info",
        "description": "Get detailed information about a specific object in the Blender scene.",
        "input_schema": {
            "type": "object",
            "properties": {
                "object_name": {
                    "type": "string",
                    "description": "The name of the object to get information about.",
                }
            },
            "required": ["object_name"],
        },
    },
    {
        "name": "get_viewport_screenshot",
        "description": "Capture a screenshot of the current Blender 3D viewport and return it as base64-encoded PNG data.",
        "input_schema": {
            "type": "object",
            "properties": {
                "max_size": {
                    "type": "integer",
                    "description": "Maximum size in pixels for the largest dimension (default: 800).",
                }
            },
            "required": [],
        },
    },
    {
        "name": "execute_blender_code",
        "description": (
            "Execute arbitrary Python code in Blender. "
            "Break complex tasks into smaller, incremental steps."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "The Python code to execute inside Blender.",
                }
            },
            "required": ["code"],
        },
    },
    {
        "name": "get_polyhaven_status",
        "description": "Check if PolyHaven integration is enabled in Blender.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_polyhaven_categories",
        "description": "Get a list of categories for a specific asset type on Polyhaven.",
        "input_schema": {
            "type": "object",
            "properties": {
                "asset_type": {
                    "type": "string",
                    "description": "Type of asset: hdris, textures, models, or all (default: hdris).",
                }
            },
            "required": [],
        },
    },
    {
        "name": "search_polyhaven_assets",
        "description": "Search for assets on Polyhaven with optional filtering.",
        "input_schema": {
            "type": "object",
            "properties": {
                "asset_type": {
                    "type": "string",
                    "description": "Type of assets: hdris, textures, models, or all (default: all).",
                },
                "categories": {
                    "type": "string",
                    "description": "Optional comma-separated list of categories to filter by.",
                },
            },
            "required": [],
        },
    },
    {
        "name": "download_polyhaven_asset",
        "description": "Download and import a Polyhaven asset into Blender.",
        "input_schema": {
            "type": "object",
            "properties": {
                "asset_id": {"type": "string", "description": "The ID of the asset to download."},
                "asset_type": {
                    "type": "string",
                    "description": "The type of asset: hdris, textures, or models.",
                },
                "resolution": {
                    "type": "string",
                    "description": "Resolution to download, e.g. 1k, 2k, 4k (default: 1k).",
                },
                "file_format": {
                    "type": "string",
                    "description": "Optional file format (e.g. hdr, exr, jpg, png, gltf, fbx).",
                },
            },
            "required": ["asset_id", "asset_type"],
        },
    },
    {
        "name": "set_texture",
        "description": "Apply a previously downloaded Polyhaven texture to an object.",
        "input_schema": {
            "type": "object",
            "properties": {
                "object_name": {
                    "type": "string",
                    "description": "Name of the object to apply the texture to.",
                },
                "texture_id": {
                    "type": "string",
                    "description": "ID of the Polyhaven texture (must be downloaded first).",
                },
            },
            "required": ["object_name", "texture_id"],
        },
    },
    {
        "name": "get_hyper3d_status",
        "description": "Check if Hyper3D Rodin integration is enabled in Blender.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_sketchfab_status",
        "description": "Check if Sketchfab integration is enabled in Blender.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "search_sketchfab_models",
        "description": "Search for models on Sketchfab with optional filtering.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Text to search for."},
                "categories": {
                    "type": "string",
                    "description": "Optional comma-separated list of categories.",
                },
                "count": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default: 20).",
                },
                "downloadable": {
                    "type": "boolean",
                    "description": "Only include downloadable models (default: true).",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_sketchfab_model_preview",
        "description": "Get a preview thumbnail of a Sketchfab model by its UID as base64-encoded image data.",
        "input_schema": {
            "type": "object",
            "properties": {
                "uid": {
                    "type": "string",
                    "description": "The unique identifier of the Sketchfab model.",
                }
            },
            "required": ["uid"],
        },
    },
    {
        "name": "download_sketchfab_model",
        "description": (
            "Download and import a Sketchfab model by its UID. "
            "The model will be scaled so its largest dimension equals target_size."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "uid": {
                    "type": "string",
                    "description": "The unique identifier of the Sketchfab model.",
                },
                "target_size": {
                    "type": "number",
                    "description": (
                        "Target size in Blender units/meters for the largest dimension. "
                        "Examples: chair=1.0, car=4.5, person=1.7, cup=0.1."
                    ),
                },
            },
            "required": ["uid", "target_size"],
        },
    },
    {
        "name": "generate_hyper3d_model_via_text",
        "description": "Generate a 3D asset using Hyper3D Rodin from a text description and import it into Blender.",
        "input_schema": {
            "type": "object",
            "properties": {
                "text_prompt": {
                    "type": "string",
                    "description": "Short English description of the desired model.",
                },
                "bbox_condition": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Optional list of 3 floats controlling [Length, Width, Height] ratio.",
                },
            },
            "required": ["text_prompt"],
        },
    },
    {
        "name": "generate_hyper3d_model_via_images",
        "description": "Generate a 3D asset using Hyper3D Rodin from input images and import it into Blender.",
        "input_schema": {
            "type": "object",
            "properties": {
                "input_image_paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Absolute file paths of input images (for MAIN_SITE mode).",
                },
                "input_image_urls": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "URLs of input images (for FAL_AI mode).",
                },
                "bbox_condition": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Optional list of 3 floats controlling [Length, Width, Height] ratio.",
                },
            },
            "required": [],
        },
    },
    {
        "name": "poll_rodin_job_status",
        "description": "Check if a Hyper3D Rodin generation task is completed.",
        "input_schema": {
            "type": "object",
            "properties": {
                "subscription_key": {
                    "type": "string",
                    "description": "Subscription key from the generate step (MAIN_SITE mode).",
                },
                "request_id": {
                    "type": "string",
                    "description": "Request ID from the generate step (FAL_AI mode).",
                },
            },
            "required": [],
        },
    },
    {
        "name": "import_generated_asset",
        "description": "Import the GLB asset generated by Hyper3D Rodin after generation completes.",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Name for the imported object in the scene."},
                "task_uuid": {
                    "type": "string",
                    "description": "Task UUID from the generate step (MAIN_SITE mode).",
                },
                "request_id": {
                    "type": "string",
                    "description": "Request ID from the generate step (FAL_AI mode).",
                },
            },
            "required": ["name"],
        },
    },
    {
        "name": "get_hunyuan3d_status",
        "description": "Check if Hunyuan3D integration is enabled in Blender.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "generate_hunyuan3d_model",
        "description": "Generate a 3D asset using Hunyuan3D from text or image and import it into Blender.",
        "input_schema": {
            "type": "object",
            "properties": {
                "text_prompt": {
                    "type": "string",
                    "description": "Optional short description of the desired model.",
                },
                "input_image_url": {
                    "type": "string",
                    "description": "Optional local path or URL of an input image.",
                },
            },
            "required": [],
        },
    },
    {
        "name": "poll_hunyuan_job_status",
        "description": "Check if a Hunyuan3D generation task is completed.",
        "input_schema": {
            "type": "object",
            "properties": {
                "job_id": {
                    "type": "string",
                    "description": "The job_id returned from generate_hunyuan3d_model.",
                }
            },
            "required": ["job_id"],
        },
    },
    {
        "name": "import_generated_asset_hunyuan",
        "description": "Import the OBJ asset generated by Hunyuan3D after generation completes.",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Name for the imported object in the scene."},
                "zip_file_url": {
                    "type": "string",
                    "description": "The zip_file_url returned from poll_hunyuan_job_status.",
                },
            },
            "required": ["name", "zip_file_url"],
        },
    },
]


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def _process_bbox(original_bbox):
    if original_bbox is None:
        return None
    if all(isinstance(i, int) for i in original_bbox):
        return original_bbox
    if any(i <= 0 for i in original_bbox):
        raise ValueError("Incorrect number range: bbox must be bigger than zero!")
    return [int(float(i) / max(original_bbox) * 100) for i in original_bbox]


def dispatch_tool(tool_name: str, tool_input: dict) -> str:
    """Call the appropriate Blender command for a given tool and return a string result."""
    blender = get_blender_connection()

    # ---- scene / object info -----------------------------------------------
    if tool_name == "get_scene_info":
        result = blender.send_command("get_scene_info")
        return json.dumps(result, indent=2)

    elif tool_name == "get_object_info":
        result = blender.send_command("get_object_info", {"name": tool_input["object_name"]})
        return json.dumps(result, indent=2)

    elif tool_name == "get_viewport_screenshot":
        max_size = tool_input.get("max_size", 800)
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, f"blender_screenshot_{os.getpid()}.png")
        result = blender.send_command("get_viewport_screenshot", {
            "max_size": max_size,
            "filepath": temp_path,
            "format": "png",
        })
        if "error" in result:
            return f"Error capturing screenshot: {result['error']}"
        if not os.path.exists(temp_path):
            return "Error: Screenshot file was not created."
        with open(temp_path, "rb") as f:
            image_bytes = f.read()
        os.remove(temp_path)
        encoded = base64.b64encode(image_bytes).decode("ascii")
        return f"Screenshot captured successfully (base64 PNG, {len(image_bytes)} bytes):\n{encoded}"

    elif tool_name == "execute_blender_code":
        result = blender.send_command("execute_code", {"code": tool_input["code"]})
        return f"Code executed successfully: {result.get('result', '')}"

    # ---- PolyHaven ----------------------------------------------------------
    elif tool_name == "get_polyhaven_status":
        result = blender.send_command("get_polyhaven_status")
        enabled = result.get("enabled", False)
        message = result.get("message", "")
        if enabled:
            message += " PolyHaven is good at Textures, and has a wider variety of textures than Sketchfab."
        return message

    elif tool_name == "get_polyhaven_categories":
        asset_type = tool_input.get("asset_type", "hdris")
        result = blender.send_command("get_polyhaven_categories", {"asset_type": asset_type})
        if "error" in result:
            return f"Error: {result['error']}"
        categories = result["categories"]
        formatted = f"Categories for {asset_type}:\n\n"
        for category, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
            formatted += f"- {category}: {count} assets\n"
        return formatted

    elif tool_name == "search_polyhaven_assets":
        params = {
            "asset_type": tool_input.get("asset_type", "all"),
            "categories": tool_input.get("categories"),
        }
        result = blender.send_command("search_polyhaven_assets", params)
        if "error" in result:
            return f"Error: {result['error']}"
        assets = result["assets"]
        total_count = result["total_count"]
        returned_count = result["returned_count"]
        categories = tool_input.get("categories")
        formatted = f"Found {total_count} assets"
        if categories:
            formatted += f" in categories: {categories}"
        formatted += f"\nShowing {returned_count} assets:\n\n"
        for asset_id, asset_data in sorted(
            assets.items(), key=lambda x: x[1].get("download_count", 0), reverse=True
        ):
            formatted += f"- {asset_data.get('name', asset_id)} (ID: {asset_id})\n"
            formatted += f"  Type: {['HDRI', 'Texture', 'Model'][asset_data.get('type', 0)]}\n"
            formatted += f"  Categories: {', '.join(asset_data.get('categories', []))}\n"
            formatted += f"  Downloads: {asset_data.get('download_count', 'Unknown')}\n\n"
        return formatted

    elif tool_name == "download_polyhaven_asset":
        result = blender.send_command("download_polyhaven_asset", {
            "asset_id": tool_input["asset_id"],
            "asset_type": tool_input["asset_type"],
            "resolution": tool_input.get("resolution", "1k"),
            "file_format": tool_input.get("file_format"),
        })
        if "error" in result:
            return f"Error: {result['error']}"
        if result.get("success"):
            message = result.get("message", "Asset downloaded and imported successfully")
            asset_type = tool_input["asset_type"]
            if asset_type == "hdris":
                return f"{message}. The HDRI has been set as the world environment."
            elif asset_type == "textures":
                material_name = result.get("material", "")
                maps = ", ".join(result.get("maps", []))
                return f"{message}. Created material '{material_name}' with maps: {maps}."
            elif asset_type == "models":
                return f"{message}. The model has been imported into the current scene."
            return message
        return f"Failed to download asset: {result.get('message', 'Unknown error')}"

    elif tool_name == "set_texture":
        result = blender.send_command("set_texture", {
            "object_name": tool_input["object_name"],
            "texture_id": tool_input["texture_id"],
        })
        if "error" in result:
            return f"Error: {result['error']}"
        if result.get("success"):
            material_name = result.get("material", "")
            maps = ", ".join(result.get("maps", []))
            material_info = result.get("material_info", {})
            node_count = material_info.get("node_count", 0)
            has_nodes = material_info.get("has_nodes", False)
            texture_nodes = material_info.get("texture_nodes", [])
            output = (
                f"Successfully applied texture '{tool_input['texture_id']}' to "
                f"{tool_input['object_name']}.\n"
                f"Using material '{material_name}' with maps: {maps}.\n\n"
                f"Material has nodes: {has_nodes}\n"
                f"Total node count: {node_count}\n\n"
            )
            if texture_nodes:
                output += "Texture nodes:\n"
                for node in texture_nodes:
                    output += f"- {node['name']} using image: {node['image']}\n"
                    if node.get("connections"):
                        output += "  Connections:\n"
                        for conn in node["connections"]:
                            output += f"    {conn}\n"
            else:
                output += "No texture nodes found in the material.\n"
            return output
        return f"Failed to apply texture: {result.get('message', 'Unknown error')}"

    # ---- Hyper3D / Rodin ----------------------------------------------------
    elif tool_name == "get_hyper3d_status":
        result = blender.send_command("get_hyper3d_status")
        return result.get("message", "")

    elif tool_name == "generate_hyper3d_model_via_text":
        result = blender.send_command("create_rodin_job", {
            "text_prompt": tool_input["text_prompt"],
            "images": None,
            "bbox_condition": _process_bbox(tool_input.get("bbox_condition")),
        })
        if result.get("submit_time"):
            return json.dumps({
                "task_uuid": result["uuid"],
                "subscription_key": result["jobs"]["subscription_key"],
            })
        return json.dumps(result)

    elif tool_name == "generate_hyper3d_model_via_images":
        input_image_paths = tool_input.get("input_image_paths")
        input_image_urls = tool_input.get("input_image_urls")
        if input_image_paths and input_image_urls:
            return "Error: Conflicting parameters – provide either input_image_paths or input_image_urls, not both."
        if not input_image_paths and not input_image_urls:
            return "Error: No image given!"
        images = None
        if input_image_paths:
            if not all(os.path.exists(p) for p in input_image_paths):
                return "Error: Not all image paths are valid!"
            images = []
            for path in input_image_paths:
                with open(path, "rb") as f:
                    images.append((Path(path).suffix, base64.b64encode(f.read()).decode("ascii")))
        elif input_image_urls:
            if not all(urlparse(u).scheme for u in input_image_urls):
                return "Error: Not all image URLs are valid!"
            images = list(input_image_urls)
        result = blender.send_command("create_rodin_job", {
            "text_prompt": None,
            "images": images,
            "bbox_condition": _process_bbox(tool_input.get("bbox_condition")),
        })
        if result.get("submit_time"):
            return json.dumps({
                "task_uuid": result["uuid"],
                "subscription_key": result["jobs"]["subscription_key"],
            })
        return json.dumps(result)

    elif tool_name == "poll_rodin_job_status":
        kwargs = {}
        if tool_input.get("subscription_key"):
            kwargs["subscription_key"] = tool_input["subscription_key"]
        elif tool_input.get("request_id"):
            kwargs["request_id"] = tool_input["request_id"]
        result = blender.send_command("poll_rodin_job_status", kwargs)
        return json.dumps(result) if isinstance(result, dict) else str(result)

    elif tool_name == "import_generated_asset":
        kwargs = {"name": tool_input["name"]}
        if tool_input.get("task_uuid"):
            kwargs["task_uuid"] = tool_input["task_uuid"]
        elif tool_input.get("request_id"):
            kwargs["request_id"] = tool_input["request_id"]
        result = blender.send_command("import_generated_asset", kwargs)
        return json.dumps(result) if isinstance(result, dict) else str(result)

    # ---- Sketchfab ----------------------------------------------------------
    elif tool_name == "get_sketchfab_status":
        result = blender.send_command("get_sketchfab_status")
        enabled = result.get("enabled", False)
        message = result.get("message", "")
        if enabled:
            message += " Sketchfab is good at Realistic models, and has a wider variety of models than PolyHaven."
        return message

    elif tool_name == "search_sketchfab_models":
        result = blender.send_command("search_sketchfab_models", {
            "query": tool_input["query"],
            "categories": tool_input.get("categories"),
            "count": tool_input.get("count", 20),
            "downloadable": tool_input.get("downloadable", True),
        })
        if result is None:
            return "Error: Received no response from Sketchfab search."
        if "error" in result:
            return f"Error: {result['error']}"
        models = result.get("results", []) or []
        if not models:
            return f"No models found matching '{tool_input['query']}'."
        formatted = f"Found {len(models)} models matching '{tool_input['query']}':\n\n"
        for model in models:
            if model is None:
                continue
            formatted += f"- {model.get('name', 'Unnamed')} (UID: {model.get('uid', 'Unknown')})\n"
            user = model.get("user") or {}
            formatted += f"  Author: {user.get('username', 'Unknown') if isinstance(user, dict) else 'Unknown'}\n"
            license_data = model.get("license") or {}
            formatted += f"  License: {license_data.get('label', 'Unknown') if isinstance(license_data, dict) else 'Unknown'}\n"
            formatted += f"  Face count: {model.get('faceCount', 'Unknown')}\n"
            formatted += f"  Downloadable: {'Yes' if model.get('isDownloadable') else 'No'}\n\n"
        return formatted

    elif tool_name == "get_sketchfab_model_preview":
        result = blender.send_command("get_sketchfab_model_preview", {"uid": tool_input["uid"]})
        if result is None:
            return "Error: Received no response from Blender."
        if "error" in result:
            return f"Error: {result['error']}"
        model_name = result.get("model_name", "Unknown")
        author = result.get("author", "Unknown")
        img_format = result.get("format", "jpeg")
        encoded = result.get("image_data", "")
        return (
            f"Preview for '{model_name}' by {author} ({img_format}):\n"
            f"(base64-encoded image data)\n{encoded}"
        )

    elif tool_name == "download_sketchfab_model":
        result = blender.send_command("download_sketchfab_model", {
            "uid": tool_input["uid"],
            "normalize_size": True,
            "target_size": tool_input["target_size"],
        })
        if result is None:
            return "Error: Received no response from Sketchfab download request."
        if "error" in result:
            return f"Error: {result['error']}"
        if result.get("success"):
            imported_objects = result.get("imported_objects", [])
            output = f"Successfully imported model.\nCreated objects: {', '.join(imported_objects) or 'none'}\n"
            if result.get("dimensions"):
                dims = result["dimensions"]
                output += f"Dimensions (X, Y, Z): {dims[0]:.3f} x {dims[1]:.3f} x {dims[2]:.3f} meters\n"
            if result.get("world_bounding_box"):
                bbox = result["world_bounding_box"]
                output += f"Bounding box: min={bbox[0]}, max={bbox[1]}\n"
            if result.get("normalized"):
                scale = result.get("scale_applied", 1.0)
                output += f"Size normalized: scale factor {scale:.6f} (target: {tool_input['target_size']}m)\n"
            return output
        return f"Failed to download model: {result.get('message', 'Unknown error')}"

    # ---- Hunyuan3D ----------------------------------------------------------
    elif tool_name == "get_hunyuan3d_status":
        result = blender.send_command("get_hunyuan3d_status")
        return result.get("message", "")

    elif tool_name == "generate_hunyuan3d_model":
        result = blender.send_command("create_hunyuan_job", {
            "text_prompt": tool_input.get("text_prompt"),
            "image": tool_input.get("input_image_url"),
        })
        if "JobId" in result.get("Response", {}):
            job_id = result["Response"]["JobId"]
            return json.dumps({"job_id": f"job_{job_id}"})
        return json.dumps(result)

    elif tool_name == "poll_hunyuan_job_status":
        result = blender.send_command("poll_hunyuan_job_status", {"job_id": tool_input.get("job_id")})
        return json.dumps(result) if isinstance(result, dict) else str(result)

    elif tool_name == "import_generated_asset_hunyuan":
        kwargs = {"name": tool_input["name"]}
        if tool_input.get("zip_file_url"):
            kwargs["zip_file_url"] = tool_input["zip_file_url"]
        result = blender.send_command("import_generated_asset_hunyuan", kwargs)
        return json.dumps(result) if isinstance(result, dict) else str(result)

    return f"Unknown tool: {tool_name}"


# ---------------------------------------------------------------------------
# Interactive chat loop
# ---------------------------------------------------------------------------

async def chat():
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable is not set.")

    base_url = os.getenv("ANTHROPIC_BASE_URL")
    model = os.getenv("ANTHROPIC_MODEL", DEFAULT_MODEL)

    client_kwargs = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url

    client = _Client(**client_kwargs)

    messages = []
    print("Blender AI Assistant (type 'exit' to quit)\n")

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() == "exit":
            break

        messages.append({"role": "user", "content": user_input})

        # Agentic loop: keep calling the model until it produces a final text reply.
        while True:
            response = client.messages.create(
                model=model,
                max_tokens=4096,
                tools=TOOLS,
                messages=messages,
            )

            # Collect all content blocks from the response.
            tool_uses = [c for c in response.content if c.type == "tool_use"]
            text_blocks = [c for c in response.content if c.type == "text"]

            if tool_uses:
                # Append the assistant turn with ALL content blocks.
                messages.append({"role": "assistant", "content": response.content})

                # Execute every requested tool and collect results.
                tool_results = []
                for tool_use in tool_uses:
                    print(f"\n[Tool Call] {tool_use.name} -> {tool_use.input}\n")
                    try:
                        output = dispatch_tool(tool_use.name, tool_use.input)
                    except Exception as e:
                        output = f"Tool error: {e}"
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_use.id,
                        "content": output,
                    })

                messages.append({"role": "user", "content": tool_results})
                continue  # Feed results back to the model.

            # No tool calls – this is the final text response.
            if text_blocks:
                reply = "\n".join(b.text for b in text_blocks)
                print(f"AI: {reply}\n")
                messages.append({"role": "assistant", "content": reply})
            break


if __name__ == "__main__":
    asyncio.run(chat())
