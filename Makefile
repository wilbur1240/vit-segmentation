.PHONY: dev-cpu dev-gpu clean

dev-cpu:
	ln -sfn devcontainers/.devcontainer-cpu .devcontainer
	@echo "Switched to CPU devcontainer"

dev-gpu:
	ln -sfn devcontainers/.devcontainer-gpu .devcontainer
	@echo "Switch to GPU devcontainer"

clean:
	rm -f .devcontainer
	@echo "Cleaned .devcontainer symlink"