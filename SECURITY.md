# Security policy

## Supported versions

| Version | Supported |
| --- | --- |
| 1.0.x | yes |

## Reporting a vulnerability

Please open a **private security advisory** on this repository rather than a
public issue, and allow a reasonable window before disclosure.

Include the version, the smallest input that reproduces the problem, and what
you were able to do with it.

## Scope

The toolkit executes instructions that come from configuration files and reads
data from files, databases and the network. The controls, the threat model and
the known residual risks (DNS rebinding, regex cost in rule files, path
time-of-check/time-of-use) are documented in
[docs/security.md](docs/security.md).

Things that are **not** vulnerabilities:

* Reading a pickle file after explicitly passing `allow_pickle=True` - that
  switch exists precisely because unpickling runs code.
* Reaching a private address after explicitly passing `allow_private_networks`
  or `--allow-private`.
* A rule file doing something expensive. Rule files are operator-supplied and
  should be reviewed like code.
* `read_query` being a keyword filter rather than a SQL parser. It is a guard
  against a careless configuration; the authoritative control is a read-only
  database role.
