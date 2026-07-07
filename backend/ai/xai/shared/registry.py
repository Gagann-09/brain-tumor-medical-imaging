from ai.xai.shared.base import BaseExplainer


class ExplainerRegistration:
    def __init__(
        self, name: str, explainer_class: type[BaseExplainer], supported_families: list[str]
    ):
        self.name = name
        self.explainer_class = explainer_class
        self.supported_families = supported_families


class ExplainerRegistry:
    """Registry for managing Explainable AI methods."""

    _explainers: dict[str, ExplainerRegistration] = {}  # type: ignore

    @classmethod
    def register(cls, name: str, supported_families: list[str]):
        """Decorator to register an explainer class."""

        def wrapper(explainer_class: type[BaseExplainer]):
            cls._explainers[name] = ExplainerRegistration(name, explainer_class, supported_families)
            return explainer_class

        return wrapper

    @classmethod
    def get_explainer_class(cls, name: str) -> type[BaseExplainer]:
        if name not in cls._explainers:
            raise ValueError(f"Explainer '{name}' not found in registry.")
        return cls._explainers[name].explainer_class

    @classmethod
    def list_explainers(cls) -> dict[str, list[str]]:
        """Returns a mapping of explainer names to their supported model families."""
        return {name: reg.supported_families for name, reg in cls._explainers.items()}
