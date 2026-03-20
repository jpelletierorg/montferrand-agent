import { execFileSync } from 'node:child_process';
import { existsSync, readFileSync } from 'node:fs';
import { homedir } from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const DEFAULT_ACCOUNT_ID = '1c66ab5db74cf26bca7c536832792876';
const DEFAULT_PROJECT_NAME = 'montferrand';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const repoRoot = path.resolve(__dirname, '..');
const webDir = path.join(repoRoot, 'web');
const distDir = path.join(webDir, 'dist');

function readWranglerOauthToken() {
  const home = homedir();
  const candidatePaths = [
    path.join(home, 'Library', 'Preferences', '.wrangler', 'config', 'default.toml'),
    path.join(home, '.wrangler', 'config', 'default.toml'),
    path.join(home, '.config', '.wrangler', 'config', 'default.toml'),
  ];

  for (const configPath of candidatePaths) {
    if (!existsSync(configPath)) {
      continue;
    }

    const contents = readFileSync(configPath, 'utf8');
    const match = contents.match(/^\s*oauth_token\s*=\s*"([^"]+)"\s*$/m);
    if (match) {
      return match[1];
    }
  }

  return null;
}

function requireCloudflareToken() {
  const token =
    process.env.CLOUDFLARE_API_TOKEN
    ?? process.env.CF_API_TOKEN
    ?? readWranglerOauthToken();

  if (token) {
    return token;
  }

  throw new Error(
    'No Cloudflare API token found. Set CLOUDFLARE_API_TOKEN or run `npx wrangler login` first.',
  );
}

if (!existsSync(distDir)) {
  throw new Error(`Missing build output at ${distDir}. Run \`task build:web\` first.`);
}

const accountId = process.env.CLOUDFLARE_ACCOUNT_ID ?? process.env.CF_ACCOUNT_ID ?? DEFAULT_ACCOUNT_ID;
const projectName = process.env.CLOUDFLARE_PAGES_PROJECT ?? DEFAULT_PROJECT_NAME;
const apiToken = requireCloudflareToken();

console.log(`Deploying web/dist to Cloudflare Pages project \`${projectName}\`...`);

execFileSync(
  'npx',
  [
    'wrangler',
    'pages',
    'deploy',
    'dist',
    '--project-name',
    projectName,
    '--commit-dirty=true',
  ],
  {
    cwd: webDir,
    stdio: 'inherit',
    env: {
      ...process.env,
      CLOUDFLARE_ACCOUNT_ID: accountId,
      CLOUDFLARE_API_TOKEN: apiToken,
    },
  },
);
