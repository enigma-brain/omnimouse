from omnimouse.tasks import (
    eval,
    train,
)

if __name__ == "__main__":
    # This allows running the file directly for development/testing.
    # In production, use the console scripts declared in pyproject.toml.
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "eval":
        # Run evaluation mode
        sys.argv.pop(1)  # Remove the 'eval' argument
        eval()
    else:
        # Default to train mode
        train()
