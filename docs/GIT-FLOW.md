# Git Flow Workflow Guide
## OpenCV-YOLO Project

**Version:** 1.0
**Last Updated:** 2025-10-17
**Status:** Active

---

## Overview

This project follows the **Git Flow** branching strategy for organized, scalable development. Git Flow provides a robust framework for managing releases, features, and hotfixes.

---

## Branch Structure

### Protected Branches

#### `main`
- **Purpose:** Production-ready code
- **Protection:** ✅ Protected (no direct commits)
- **Merges from:** `release/*`, `hotfix/*`
- **Tagged:** All releases tagged here (e.g., `v1.0.0`)

#### `develop`
- **Purpose:** Integration branch for ongoing development
- **Protection:** ✅ Protected (no direct commits)
- **Merges from:** `feature/*`, `release/*`, `hotfix/*`
- **Base for:** `feature/*`, `release/*`

### Working Branches

#### `feature/*`
- **Purpose:** New features and enhancements
- **Naming:** `feature/descriptive-name`
- **Base:** `develop`
- **Merges to:** `develop`
- **Lifetime:** Temporary (deleted after merge)

**Examples:**
```
feature/hungarian-algorithm
feature/kalman-filter-optimization
feature/multi-iou-support
feature/trajectory-visualization
```

#### `release/*`
- **Purpose:** Release preparation and final testing
- **Naming:** `release/vX.Y.Z`
- **Base:** `develop`
- **Merges to:** `main` AND `develop`
- **Tagged:** Yes (on `main`)
- **Lifetime:** Temporary (deleted after merge)

**Examples:**
```
release/v1.0.0
release/v1.1.0-beta
release/v2.0.0
```

#### `hotfix/*`
- **Purpose:** Emergency fixes for production issues
- **Naming:** `hotfix/descriptive-name` or `hotfix/vX.Y.Z`
- **Base:** `main`
- **Merges to:** `main` AND `develop`
- **Tagged:** Yes (on `main`)
- **Lifetime:** Temporary (deleted after merge)

**Examples:**
```
hotfix/memory-leak-fix
hotfix/v1.0.1
hotfix/tracking-crash
```

---

## Workflow Diagrams

### Feature Development Flow
```
develop ──┐
          │
          ├─> feature/new-feature (create)
          │         │
          │         ├─> commits...
          │         │
          │         └─> ready to merge
          │
          └─> merge back to develop
```

### Release Flow
```
develop ──┐
          │
          ├─> release/v1.0.0 (create)
          │         │
          │         ├─> version bump
          │         ├─> bug fixes
          │         ├─> documentation
          │         │
          ├─────────┴─> merge to main (tag v1.0.0)
          │
          └─> merge back to develop
```

### Hotfix Flow
```
main ──┐
       │
       ├─> hotfix/critical-bug (create)
       │         │
       │         ├─> fix commits
       │         │
       ├─────────┴─> merge to main (tag v1.0.1)
       │
develop└─> merge to develop
```

---

## Command Reference

### Starting New Work

#### Create Feature Branch
```bash
# Ensure you're on develop
git checkout develop
git pull origin develop

# Create feature branch
git checkout -b feature/my-new-feature

# Push to remote and set tracking
git push -u origin feature/my-new-feature
```

#### Create Release Branch
```bash
# From develop
git checkout develop
git pull origin develop

# Create release branch
git checkout -b release/v1.2.0

# Update version in relevant files
# Commit version bump
git commit -am "chore(release): bump version to 1.2.0"

# Push to remote
git push -u origin release/v1.2.0
```

#### Create Hotfix Branch
```bash
# From main
git checkout main
git pull origin main

# Create hotfix branch
git checkout -b hotfix/critical-fix

# Push to remote
git push -u origin hotfix/critical-fix
```

---

### Finishing Work

#### Finish Feature (Manual)
```bash
# Ensure feature is complete and tested
git checkout develop
git pull origin develop

# Merge feature (no fast-forward for history clarity)
git merge --no-ff feature/my-new-feature

# Push to develop
git push origin develop

# Delete local and remote branches
git branch -d feature/my-new-feature
git push origin --delete feature/my-new-feature
```

#### Finish Release (Manual)
```bash
# Merge to main first
git checkout main
git pull origin main
git merge --no-ff release/v1.2.0

# Tag the release
git tag -a v1.2.0 -m "Release version 1.2.0"
git push origin main --tags

# Merge back to develop
git checkout develop
git pull origin develop
git merge --no-ff release/v1.2.0
git push origin develop

# Delete release branch
git branch -d release/v1.2.0
git push origin --delete release/v1.2.0
```

#### Finish Hotfix (Manual)
```bash
# Merge to main
git checkout main
git pull origin main
git merge --no-ff hotfix/critical-fix

# Tag the hotfix
git tag -a v1.0.1 -m "Hotfix v1.0.1: Critical bug fix"
git push origin main --tags

# Merge to develop
git checkout develop
git pull origin develop
git merge --no-ff hotfix/critical-fix
git push origin develop

# Delete hotfix branch
git branch -d hotfix/critical-fix
git push origin --delete hotfix/critical-fix
```

---

### Using Slash Commands (Recommended)

The project includes Git Flow slash commands for automation:

```bash
# Create feature branch
/gitFlow:feature my-new-feature

# Create release branch
/gitFlow:release v1.2.0

# Create hotfix branch
/gitFlow:hotfix critical-fix

# Check Git Flow status
/gitFlow:flow-status

# Finish current branch (automatic detection)
/gitFlow:finish

# Finish without deleting branch
/gitFlow:finish --no-delete

# Finish release without tagging
/gitFlow:finish --no-tag
```

---

## Commit Message Convention

This project follows **Conventional Commits** specification:

### Format
```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

### Types
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, no logic change)
- `refactor`: Code refactoring (no feature/fix)
- `perf`: Performance improvements
- `test`: Adding or updating tests
- `chore`: Build process, tooling, dependencies

### Examples
```bash
# Feature
git commit -m "feat(tracking): add Hungarian algorithm for optimal assignment"

# Bug fix
git commit -m "fix(kalman): correct prediction matrix initialization"

# Documentation
git commit -m "docs(readme): update build instructions for CMake"

# Performance
git commit -m "perf(inference): optimize CUDA memory allocation"

# Breaking change
git commit -m "feat(api)!: change KalmanIoUTracker initialization signature

BREAKING CHANGE: KalmanIoUTracker now requires ImpressionSettings in constructor"
```

---

## Semantic Versioning

Project follows **Semantic Versioning 2.0.0** (`vMAJOR.MINOR.PATCH`):

### Version Increments

| Change Type | Example | Version Bump |
|-------------|---------|--------------|
| Breaking changes | API signature change | `v1.0.0` → `v2.0.0` |
| New features | Add SIOU algorithm | `v1.0.0` → `v1.1.0` |
| Bug fixes | Fix memory leak | `v1.0.0` → `v1.0.1` |
| Pre-release | Beta version | `v1.0.0-beta.1` |

### Current Version
**Version:** `v0.1.0` (Initial Development)

**Changelog:**
- `v0.1.0` (2025-10-17): Initial Git Flow setup
  - Established branch structure
  - Added documentation
  - Organized project structure

---

## Best Practices

### DO ✅

1. **Always branch from correct base**
   - Features from `develop`
   - Releases from `develop`
   - Hotfixes from `main`

2. **Pull before creating branches**
   ```bash
   git pull origin develop  # or main for hotfixes
   ```

3. **Use descriptive branch names**
   - Good: `feature/multi-object-tracking`
   - Bad: `feature/update`

4. **Commit frequently with clear messages**
   - Small, atomic commits
   - Descriptive commit messages

5. **Test before finishing branches**
   - Run build: `cmake --build build/`
   - Run tests (if available)
   - Verify functionality

6. **Use --no-ff merges for clarity**
   ```bash
   git merge --no-ff feature/branch-name
   ```

7. **Tag all releases and hotfixes**
   ```bash
   git tag -a v1.0.0 -m "Release v1.0.0"
   ```

8. **Delete branches after merge**
   ```bash
   git branch -d feature/completed-feature
   git push origin --delete feature/completed-feature
   ```

### DON'T ❌

1. **Never commit directly to `main` or `develop`**
   ```bash
   # BAD
   git checkout main
   git commit -m "quick fix"  # ❌ NO!
   ```

2. **Never force push to shared branches**
   ```bash
   git push --force origin develop  # ❌ NEVER!
   ```

3. **Avoid long-lived feature branches**
   - Merge frequently to avoid conflicts
   - Keep features small and focused

4. **Don't skip testing before merge**
   - Always build and test before finishing

5. **Never merge without pulling first**
   ```bash
   git pull origin develop  # Always do this first
   git merge feature/branch
   ```

6. **Avoid generic commit messages**
   - Bad: `git commit -m "update"`
   - Good: `git commit -m "feat(inference): add YOLOv11 support"`

---

## Common Scenarios

### Starting a New Feature

```bash
# 1. Sync develop
git checkout develop
git pull origin develop

# 2. Create feature branch
git checkout -b feature/add-bytetrack-algorithm

# 3. Work on feature
# ... make changes ...
git add .
git commit -m "feat(tracking): implement ByteTrack algorithm"

# 4. Push regularly
git push origin feature/add-bytetrack-algorithm

# 5. When complete, create PR or merge manually
```

### Preparing a Release

```bash
# 1. Create release branch
git checkout develop
git pull origin develop
git checkout -b release/v1.0.0

# 2. Update version numbers
# Edit CMakeLists.txt, documentation, etc.
git commit -am "chore(release): bump version to 1.0.0"

# 3. Final testing and bug fixes
git commit -am "fix(release): resolve edge case in tracker"

# 4. Finish release (manual or use /gitFlow:finish)
# Merge to main and develop, tag, delete branch
```

### Emergency Hotfix

```bash
# 1. Create from main
git checkout main
git pull origin main
git checkout -b hotfix/segfault-in-hungarian

# 2. Fix the issue
git commit -am "fix(hungarian): prevent segfault with empty detection list"

# 3. Finish hotfix immediately
# Merge to main (tag v1.0.1) and develop
```

---

## Conflict Resolution

### If Conflicts Occur During Merge

```bash
# 1. Attempt merge
git merge --no-ff feature/branch-name
# CONFLICT message appears

# 2. Check conflicting files
git status

# 3. Open files, resolve conflicts
# Look for <<<<<<< HEAD markers
# Edit to keep desired changes

# 4. Mark as resolved
git add resolved-file.cpp

# 5. Complete merge
git commit
```

### Prevention Tips
- Merge `develop` into your feature branch regularly
- Keep feature branches short-lived
- Communicate with team about overlapping work

---

## CI/CD Integration

### Automated Workflows (Future)

**On Pull Request to `develop`:**
- [ ] Run CMake build
- [ ] Run unit tests
- [ ] Check code formatting
- [ ] Run static analysis

**On Merge to `main`:**
- [ ] Build release artifacts
- [ ] Create GitHub release
- [ ] Tag with version number
- [ ] Deploy documentation

**On Push to `feature/*`:**
- [ ] Run basic build check
- [ ] Lint code

---

## Troubleshooting

### "Branch Already Exists"
```bash
# Delete local branch
git branch -d feature/old-branch

# Delete remote branch
git push origin --delete feature/old-branch
```

### "Cannot Delete Branch (Not Fully Merged)"
```bash
# Force delete (if sure it's merged)
git branch -D feature/old-branch
```

### "Detached HEAD State"
```bash
# Return to a branch
git checkout develop
```

### "Merge Conflict Too Complex"
```bash
# Abort merge and try again
git merge --abort

# Or use a merge tool
git mergetool
```

---

## Project-Specific Notes

### Model Files
- **All model files (`.onnx`, `.pt`) are in `.gitignore`**
- Never commit model files to the repository
- Use external storage (Git LFS, cloud storage)
- Document model sources in [CLAUDE.md](../CLAUDE.md)

### Build Artifacts
- `build/` directory is ignored
- Always clean build before release testing

### Third-Party Libraries
- `thirdParty/` is tracked in git
- Document dependencies in [CLAUDE.md](../CLAUDE.md)

---

## Quick Reference Card

| Task | Command |
|------|---------|
| Create feature | `git checkout -b feature/name` |
| Create release | `git checkout -b release/vX.Y.Z` |
| Create hotfix | `git checkout -b hotfix/name` |
| Finish feature | Merge to `develop` |
| Finish release | Merge to `main` & `develop`, tag |
| Finish hotfix | Merge to `main` & `develop`, tag |
| Check status | `git status`, `/gitFlow:flow-status` |
| View branches | `git branch -a` |
| Delete branch | `git branch -d name` |
| Delete remote | `git push origin --delete name` |

---

## Resources

### Git Flow Resources
- [Git Flow Cheat Sheet](https://danielkummer.github.io/git-flow-cheatsheet/)
- [Atlassian Git Flow Tutorial](https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow)
- [Conventional Commits](https://www.conventionalcommits.org/)
- [Semantic Versioning](https://semver.org/)

### Project Documentation
- [CLAUDE.md](../CLAUDE.md) - Project instructions for Claude Code
- [explain-architecture-pattern.md](explain-architecture-pattern.md) - Architecture documentation

---

## Changelog

### v1.0.0 (2025-10-17)
- Initial Git Flow documentation
- Established branch naming conventions
- Defined workflow procedures
- Created slash commands for automation

---

**Maintained By:** Development Team
**Questions?** See [CLAUDE.md](../CLAUDE.md) or create an issue

---

## Appendix: Visual Workflow

```
                    main (production)
                      │
                      │  (hotfix) ─┐
                      │            │
                      ▼            │
┌─────────────────────────────────┴───────┐
│         Tagged Releases                 │
│     v1.0.0    v1.1.0    v1.0.1         │
└─────────────────────────────────────────┘
                      │
                      │ (release)
                      │
                   develop (integration)
                      │
        ┌─────────────┼─────────────┐
        │             │             │
   feature/a     feature/b     feature/c
```

**Remember:** Git Flow brings structure to team collaboration. Follow the conventions and keep branches clean!
