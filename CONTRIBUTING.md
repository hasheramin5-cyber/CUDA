# Contributing to CUDA

Thank you for your interest in contributing to this repository.

This project is a collection of practical CUDA projects designed to help students, learners, and developers understand and work with GPU computing through real, runnable examples.

Contributions are welcome, including new CUDA projects, bug fixes, performance improvements, documentation updates, and CI improvements.

## Before Contributing

Before making changes:

1. Read the repository `README.md`.
2. Check existing projects to avoid unnecessary duplication.
3. Make sure your contribution fits the purpose of this repository.
4. Keep changes focused and easy to review.

## Repository Structure

Each CUDA project should follow this structure:

```text
XX. Project Name/
├── MainProgram.cu
└── README.md
```

Do not add generated binaries, build directories, or temporary files to the repository.

## Adding a New CUDA Project

When adding a new project:

1. Create a new numbered project directory.
2. Add the main CUDA source file as `MainProgram.cu`.
3. Add a `README.md` explaining the project.
4. Include proper CUDA error handling where appropriate.
5. Verify results where practical.
6. Do not commit compiled binaries or generated files.

Example:

```text
11. Device Query/
├── MainProgram.cu
└── README.md
```

## CUDA Code Guidelines

Keep CUDA programs:

- Clear and readable
- Practical and focused
- Properly structured
- Safe with CUDA API error handling
- Free from unnecessary complexity
- Easy for students and developers to understand

Use meaningful variable and function names.

Release CUDA resources properly, including device memory and other allocated resources.

## Documentation Guidelines

Each project should have a `README.md` containing, where applicable:

- Project description
- CUDA concepts used
- Requirements
- Compilation instructions
- Run instructions
- Expected output
- Explanation of how the program works
- Project structure
- GPU requirements

Do not claim that a program was tested on hardware when it was not actually tested.

## Testing

Before submitting a pull request, test your changes when you have access to a CUDA-capable NVIDIA GPU and the required CUDA Toolkit.

Useful commands include:

```bash
nvcc --version
```

```bash
nvidia-smi
```

If you cannot run CUDA locally because you do not have compatible NVIDIA hardware, clearly mention this in your pull request.

The repository CI also performs repository-level validation.

## Commit Guidelines

Use clear and descriptive commit messages.

Examples:

```text
feat: add device query CUDA project
fix: correct matrix multiplication indexing
docs: improve convolution project README
ci: improve repository validation
```

Keep each commit focused on a specific change when possible.

## Pull Requests

Before opening a pull request:

- Make sure your changes are focused.
- Check that no generated files are included.
- Update documentation when necessary.
- Test the project when possible.
- Review your changes before submitting.
- Complete the pull request template.

For CUDA-specific problems, include your:

- Operating system
- NVIDIA GPU
- CUDA Toolkit version
- CUDA Driver version
- `nvcc --version` output
- `nvidia-smi` output
- Complete error message

## Reporting Bugs

Use the appropriate issue template when reporting a problem:

- **Bug Report** - General project problems
- **CUDA Error** - CUDA, `nvcc`, driver, GPU, memory, or kernel errors
- **Feature Request** - New features, projects, or improvements

Provide enough information for contributors to reproduce and understand the issue.

## Code Review

Pull requests may be reviewed for:

- Correctness
- CUDA API usage
- Code quality
- Error handling
- Performance
- Documentation
- Repository structure
- Maintainability

Constructive feedback is encouraged.

## Questions and Discussions

If you are unsure about an implementation or want to discuss an idea before contributing, open an appropriate discussion or issue.

Students and beginners are welcome to ask questions and learn from the projects.

## License

By contributing to this repository, you agree that your contributions will be licensed under the same license as the project.

See the `LICENSE` file for details.

Thank you for contributing to the CUDA community.
