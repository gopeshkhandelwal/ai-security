# Contributing to Platform Security

This guide is for maintainers and contributors who need to build and publish the ai-security Docker image.

## Prerequisites

- Docker installed and running
- Access to the Intel container registry (`amr-registry.caas.intel.com`)
- Registry login: `docker login amr-registry.caas.intel.com`

## Building the Docker Image

**Important:** Run all commands in the same terminal session to preserve environment variables.

```bash
cd platform-security

# Set version (increment for each release)
export AI_SECURITY_VERSION="1.7.0"
export AI_SECURITY_IMAGE="amr-registry.caas.intel.com/intelcloud/ai-security:${AI_SECURITY_VERSION}"

# Verify variable is set
echo "Building: $AI_SECURITY_IMAGE"

# Build with proxy (required for corporate networks)
docker build \
  --build-arg http_proxy="$http_proxy" \
  --build-arg https_proxy="$https_proxy" \
  --build-arg no_proxy="$no_proxy" \
  -t $AI_SECURITY_IMAGE .

# Test the image
docker run $AI_SECURITY_IMAGE --help
docker run $AI_SECURITY_IMAGE scan --help
```

## Publishing the Docker Image

**Note:** The `$AI_SECURITY_IMAGE` variable must be set in the same terminal session. If you get `docker push requires 1 argument`, the variable is not set.

```bash
# Verify variable is set
echo "Pushing: $AI_SECURITY_IMAGE"

# Push versioned tag
docker push $AI_SECURITY_IMAGE
```

## Versioning

Use [Semantic Versioning](https://semver.org/) (SemVer):

| Tag | Example | Purpose |
|-----|---------|---------|
| `MAJOR.MINOR.PATCH` | `1.6.0` | Immutable release (use in production) |
| `MAJOR.MINOR` | `1.6` | Latest patch in minor version |

**Version increments:**
- **PATCH** (`1.6.0` → `1.6.1`): Bug fixes, no API changes
- **MINOR** (`1.6.0` → `1.7.0`): New features, backward compatible
- **MAJOR** (`1.6.0` → `2.0.0`): Breaking changes

**Before releasing a new version:**
1. Run the full test suite
2. Update version in documentation
3. Create a git tag for the release

## Running Tests

```bash
# Unit tests
cd platform-security/model-security
python -m pytest tests/

# E2E test
./tests/test_e2e_model_security.sh --mock
./tests/test_e2e_model_security.sh openai-community/gpt2
```

## Code Review Checklist

- [ ] No hardcoded credentials or tokens
- [ ] No `--privileged` in Docker commands
- [ ] No `chmod 777` (use proper `chown` instead)
- [ ] POSIX-compliant shell scripts (`=` not `==` in `[ ]`)
- [ ] Command substitutions quoted: `"$(id -u):$(id -g)"`
- [ ] All subprocess calls use list args (no `shell=True`)

## Security Guidelines

This is a security framework. All contributions must:

1. **Never introduce security vulnerabilities**
2. **Follow least-privilege principles**
3. **Use safe defaults**
4. **Log security-relevant events**
5. **Fail closed** (deny on error)
