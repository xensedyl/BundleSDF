# Contributing to BundleSDF

Thank you for your interest in contributing to BundleSDF! This document
explains the requirements that apply to every contribution.

BundleSDF is licensed under the [Apache License, Version 2.0](LICENSE.txt).
By contributing, you agree that your contributions will be licensed under
the same terms.

## Signing Your Work

We require that every commit be **signed off** in accordance with the
[Developer Certificate of Origin (DCO)](https://developercertificate.org/).
The DCO is a lightweight, well-established mechanism (used by the Linux
kernel and many Apache-2.0 projects) that lets you certify that you wrote
or otherwise have the right to submit the code you are contributing.

Apache-2.0 projects in this organization use the DCO; we do **not** use a
separate Contributor License Agreement (CLA). The full DCO text is
reproduced verbatim at the bottom of this file.

### How to sign off a commit

Use `git commit -s` (or `--signoff`). Git will append a `Signed-off-by:`
trailer to the commit message containing the name and email from your
`user.name` and `user.email` git configuration. The trailer must match
the identity you actually wish to certify under.

Example:

```
$ git commit -s -m "Fix off-by-one in keyframe selection"
```

This produces a commit message of the form:

```
Fix off-by-one in keyframe selection

Signed-off-by: Jane Doe <jane.doe@example.com>
```

If you forget to sign off and have not yet pushed, you can amend the
last commit:

```
$ git commit --amend -s
```

To sign off a series of commits that have already been made, you can
rebase and add sign-offs interactively, or use:

```
$ git rebase --signoff <base>
```

Pull requests whose commits are not signed off will be asked to add the
`Signed-off-by:` trailer before they can be merged.

### What signing off certifies

By signing off, you certify the four points (a) through (d) in the DCO
text reproduced below — in short: that you have the right to submit the
contribution under this project's open source license, and that you
understand the contribution and your sign-off are public and will be
maintained indefinitely.

---

## Developer Certificate of Origin (verbatim)

```
Developer Certificate of Origin
Version 1.1

Copyright (C) 2004, 2006 The Linux Foundation and its contributors.

Everyone is permitted to copy and distribute verbatim copies of this
license document, but changing it is not allowed.


Developer's Certificate of Origin 1.1

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I
    have the right to submit it under the open source license
    indicated in the file; or

(b) The contribution is based upon previous work that, to the best
    of my knowledge, is covered under an appropriate open source
    license and I have the right under that license to submit that
    work with modifications, whether created in whole or in part
    by me, under the same open source license (unless I am
    permitted to submit under a different license), as indicated
    in the file; or

(c) The contribution was provided directly to me by some other
    person who certified (a), (b) or (c) and I have not modified
    it.

(d) I understand and agree that this project and the contribution
    are public and that a record of the contribution (including all
    personal information I submit with it, including my sign-off) is
    maintained indefinitely and may be redistributed consistent with
    this project or the open source license(s) involved.
```
