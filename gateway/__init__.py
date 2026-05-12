"""Session-aware OpenAI-compatible API gateway."""

from gateway.config import GatewayConfig, LLMRouteConfig, load_gateway_config

__all__ = ["GatewayConfig", "LLMRouteConfig", "create_app", "load_gateway_config"]


def create_app(*args, **kwargs):
    from gateway.app import create_app as _create_app

    return _create_app(*args, **kwargs)
