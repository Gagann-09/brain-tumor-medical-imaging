# Contributing to ARM GAN Platform

First off, thank you for considering contributing to our medical AI platform! It's people like you that make it such a great tool.

## General Guidelines

1. **Patient Data Privacy:** NEVER commit Protected Health Information (PHI) or Personally Identifiable Information (PII). Ensure all datasets used for testing are thoroughly anonymized.
2. **Security First:** Review `SECURITY.md` before submitting code that touches data pipelines, authentication, or infrastructure.
3. **Coding Standards:** Follow the PEP 8 style guide for Python, and standard ESLint configurations for frontend code. Use `pre-commit` to format your code before submitting a PR.
4. **Testing:** Write unit and integration tests for new features. We maintain a high coverage threshold.

## Development Workflow

1. Fork the repo and create your branch from `develop`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Make sure your code lints.
6. Issue a Pull Request.

## Pull Request Process

1. Ensure any install or build dependencies are removed before the end of the layer when doing a build.
2. Update the README.md with details of changes to the interface, this includes new environment variables, exposed ports, useful file locations and container parameters.
3. Increase the version numbers in any examples files and the README.md to the new version that this Pull Request would represent. The versioning scheme we use is [SemVer](http://semver.org/).
4. You may merge the Pull Request in once you have the sign-off of two other developers, or if you do not have permission to do that, you may request the second reviewer to merge it for you.
