# Contributing

Thank you for your interest in improving this project.

## Code Style
- Follow PEP8 for Python code.
- Use 2-space indentation for frontend JavaScript/React files.
- Include tests for any new feature or bug fix.
- Public function signatures must use typed models instead of raw ``dict`` values.

## Development Workflow
1. Fork the repository and create your changes on a new branch.
2. Ensure linting and tests pass before committing:
   ```bash
   OPENAI_API_KEY=dummy pytest --maxfail=1 --disable-warnings -q
   cd frontend && npm test
   ```
3. Submit a pull request describing the changes and reference any relevant tickets.

## Pull Request Guidelines
- Keep commits focused; do not bundle unrelated changes.
- Update documentation when behavior or configuration changes.
- Ensure CI passes before requesting review.

We appreciate your contributions!
