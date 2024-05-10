# Making an RLST release

To make a new release of RLST, follow the following steps:

0) If you are yet to make a release on your current computer, run `cargo login` and copy an API
   key from https://crates.io/me

1) Checkout the branch `release` and merge the branch `main` into the branch `release`:
   ```bash
   git checkout release
   git merge main
   ```

2) Update the version numbers in `Cargo.toml` and `proc-macro/Cargo.toml`.
   The version numbers have the format `[x].[y].[z]`. If you are releasing a major
   version, you should increment `[x]` and set `[y]` and `[z]` to 0.
   If you are releasing a minor version, you should increment `[y]` and set `[z]`
   to zero. If you are releasing a bugfix, you should increment `[z]`.

3) Commit you changes and push to GitHub, and check that all the tests on CI pass.

4) [Create a release on GitHub](https://github.com/linalg-rs/rlst/releases/new) from the `release` branch.
   The release tag and title should be `v[x].[y].[z]` (where `[x]`, `[y]` and `[z]` are as in step 2).
   In the "Describe this release" box, you should bullet point the main changes since the last
   release.

5) In the `proc-macro` folder:

  a) Run `cargo publish --dry-run`, then run `cargo package --list` and
     check that no unwanted extras
     have been included in the release.

  b) If everything is working as expected, run `cargo publish`. This will push the new version to
     crates.io. Note: this cannot be undone, but you can use `cargo yank` to mark a version as
     unsuitable for use.

6) (Move back to the top rlst folder.) In `Cargo.toml`, for the `rlst-proc-macro` dependency,
   change `path = "proc-macro"` to `version = "[x].[y].[z]`.

7) 

  a) Run `cargo publish --dry-run`, then run `cargo package --list` and
     check that no unwanted extras
     have been included in the release.

  b) If everything is working as expected, run `cargo publish`. This will push the new version to
     crates.io. Note: this cannot be undone, but you can use `cargo yank` to mark a version as
     unsuitable for use.

8) Open a pull request to `main` to update the version numbers in `Cargo.toml` and
   `proc-macro/Cargo.toml` to `[x].[y].[z]-dev`
