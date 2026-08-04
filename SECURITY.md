# Security Policy

## Supported versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | ✅ |

Older versions are not patched. Please upgrade to the latest release.

## Reporting a vulnerability

**Please do not open a public issue for security vulnerabilities.**

Report security issues by email to the maintainers:

- **Primary**: licheng.zheng@aijishu.com
- **Subject line**: `[jishubench SECURITY] <brief description>`

Include as much detail as possible:

- Description of the vulnerability and its potential impact.
- Steps to reproduce or a minimal proof-of-concept.
- Affected version(s) and configuration.
- Any suggested mitigations.

You will receive an acknowledgement within **3 business days** and a resolution
timeline within **10 business days**. We follow responsible disclosure: please
allow us time to ship a fix before any public disclosure.

## Scope

This project is evaluation tooling, not a network service intended for public
exposure. The primary attack surface is:

- The Host CLI reading user-controlled YAML config files.
- HTTP clients connecting to user-specified Target endpoints.
- Subprocess execution of external harnesses (`tb`, `harbor`).

Out of scope: vulnerabilities in third-party dependencies (report those to the
respective upstream projects) or in the submodules (report
to the related submodules).
