# Dependency security update

Checked on 2026-10-06 against 215 open GitHub Dependabot alerts.

| Manifest | Original alerts | Result |
| --- | ---: | --- |
| `uv.lock` | 91 | All locked versions are outside the reported vulnerable ranges. |
| `DigitRecognizer/package-lock.json` | 118 | 115 resolved; 3 remain without a published fix. |
| `DigitRecognizer/Gemfile` | 6 | Constraints now exclude all reported vulnerable versions; Bundler resolution has not been run. |

These changes are expected to address 212 of the 215 original alerts. GitHub must rescan the default branch after merge to confirm its alert count.

## Changes

- Refresh Python packages with advisories, keeping unrelated locked versions where possible. Update the paired PyTorch/torchvision versions together, and replace a yanked NumPy release. Raise the direct dependency minimums to the tested versions.
- Also update Click after an independent Python audit identified CVE-2026-7246.
- Keep Expo SDK 54 and React Native 0.81.5. Update Expo patches, align React Native tooling to 0.81.5, remove unused template dependencies, and update the community CLI to 20.2.0 or newer in the same major version.
- Override PostCSS to the patched 8.5 series, UUID to the CommonJS-compatible 11.1 series, and Metro's image-size dependency to the patched 2.0 series. Metro's image parsing and Xcode's UUID generation were checked against these overrides.
- Use the Expo Jest preset for the existing app render test and expose `test` and `typecheck` scripts. Remove an unused import that prevented lint from passing.
- Require ActiveSupport >= 7.2.3.1 within the 7.x series and concurrent-ruby >= 1.3.7 within the 1.x series. Raise the Ruby minimum to 3.1, as required by the patched ActiveSupport release.

## Remaining advisories

| Package | Locked version | Advisory | Published fix |
| --- | --- | --- | --- |
| `braces` | 3.0.3 | [GHSA-vfj7-8cjw-p6xm](https://github.com/advisories/GHSA-vfj7-8cjw-p6xm) | None at time of check |
| `node-forge` | 1.4.0 | [GHSA-86w9-cpqp-85rv](https://github.com/advisories/GHSA-86w9-cpqp-85rv) | None at time of check |
| `sprintf-js` | 1.0.3 | [GHSA-hp3w-g68c-fv3c](https://github.com/advisories/GHSA-hp3w-g68c-fv3c) | None at time of check |

`npm audit` reports 67 affected packages (62 high, 5 moderate, no critical). These include ancestor packages affected by the three unresolved underlying advisories; this is a different count from GitHub's individual alerts. Replacing these packages requires upstream changes or a separately reviewed patch.

## Validation

- Compared all 91 original Python alerts and all 118 original npm alerts against every corresponding locked version and its reported vulnerable range.
- `uv sync --locked` and `uv lock --check` passed.
- `pip-audit` against exported locked requirements found no known vulnerabilities for the local Windows platform.
- Python imports, Pillow/torchvision preprocessing, PyTorch forward/backward computation, and scikit-learn training passed.
- Flask `/health`, `/predict`, probability output, and missing-image validation passed with a synthetic model checkpoint. The trained checkpoint is not included in the repository, so prediction accuracy was not evaluated.
- `npm ci --ignore-scripts --no-audit`, TypeScript checking, and the existing Jest render test passed.
- ESLint passed with two existing inline-style warnings.
- `expo install --check` passed, and `expo export --platform android` produced a bundle.
- Metro PNG parsing and Xcode UUID generation passed with the dependency overrides.

Ruby/Bundler and an iOS native build could not be tested because Ruby and Xcode are unavailable on this Windows machine. An Android native build and physical-device checks also remain outstanding. For iOS, use Ruby >= 3.1 and run `bundle install` before building.
