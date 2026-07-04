# Architecture Governance

This document outlines the architectural guidelines and boundaries for the project.

## Layered Architecture
We follow a strict layered architecture pattern. The dependency direction must always point **inwards** (or downwards). Higher-level modules can depend on lower-level modules, but lower-level modules must NEVER import from higher-level modules.

### Permitted Dependency Direction
1. **API Layer (`backend.api`)**: External interfaces. Can depend on Services, Models, Infrastructure, and Core.
2. **Services Layer (`backend.services`)**: Business logic. Can depend on Models, Infrastructure, and Core. Cannot depend on API.
3. **Models Layer (`backend.models`)**: Data structures and ORM entities. Can depend on Core. Cannot depend on Services or API.
4. **Infrastructure Layer (`backend.infrastructure`)**: Database connections, external clients. Can depend on Core. Cannot depend on API or Services.
5. **Core Layer (`backend.core`)**: Utilities, configuration, constants. Cannot depend on any other internal package.

## Tooling
We use `import-linter` to strictly enforce these boundaries in CI.
- **Circular Imports**: Are strictly forbidden and will be detected during the build phase.
- **Dependency Violations**: Importing "up" the layer stack (e.g., Models importing from Services) will fail the build.
- **Package Boundary Violations**: Cross-domain imports must happen at the service level, never at the repository/model level.

## Semantic Versioning (SemVer)
This project strictly adheres to [Semantic Versioning (SemVer)](https://semver.org/).
- **MAJOR** version when you make incompatible API changes,
- **MINOR** version when you add functionality in a backwards compatible manner, and
- **PATCH** version when you make backwards compatible bug fixes.

All stable releases must be tagged in Git (e.g., `v0.1.0`). Any release deploying models to production must undergo the architecture and security checklist.
