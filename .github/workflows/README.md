# GitHub Actions Workflows

This directory contains automated workflows for the trading bot.

## Available Workflows

### 📊 Fetch TradingView Data

**File**: `fetch-tradingview-data.yml`

Automatically fetches historical market data for all TradingView-compatible futures symbols.

- **Schedule**: Every Sunday at 2 AM UTC (weekly)
- **Manual Trigger**: Available via GitHub Actions UI
- **Scope**: All compatible symbols, all intervals (5m to 1d)
- **Duration**: ~30-60 minutes per run
- **Output**: Commits updated data to `data/historical_data/`

**Quick Actions**:

```bash
# Trigger manually via GitHub CLI
gh workflow run fetch-tradingview-data.yml

# View recent runs
gh run list --workflow=fetch-tradingview-data.yml

# View logs
gh run view --log
```

**Configuration**: See `.github/prompts/github_actions_data_fetching.md` for detailed documentation.

## Workflow Status

Check the [Actions tab](../../actions) to view:

- ✅ Recent workflow runs
- 📋 Execution logs
- 📦 Artifacts (log files)
- 📊 Summary reports

## Setup Requirements

### Permissions

Workflows require write permissions to commit data:

1. Go to **Settings** → **Actions** → **General**
2. Select **Read and write permissions**
3. Save changes

### Secrets

No additional secrets required - uses `GITHUB_TOKEN` automatically.

## Adding New Workflows

1. Create a new `.yml` file in this directory
2. Define triggers (`on:` section)
3. Define jobs and steps
4. Test locally with [act](https://github.com/nektos/act) if possible
5. Commit and push to enable

## Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Workflow Syntax](https://docs.github.com/en/actions/reference/workflow-syntax-for-github-actions)
- [Cron Schedule Helper](https://crontab.guru/)

