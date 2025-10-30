# Emotion-LLaMA Documentation

This directory contains the source files for the Emotion-LLaMA GitHub Pages documentation site, built with the [Just the Docs](https://just-the-docs.com/) Jekyll theme.

## Building Locally

### Prerequisites

- Ruby 2.7 or higher
- Bundler

### Installation

```bash
cd docs
bundle install
```

### Local Development

```bash
bundle exec jekyll serve
```

Then visit `http://localhost:4000/Emotion-LLaMA/` in your browser.

### Live Reload

For automatic rebuilding on file changes:

```bash
bundle exec jekyll serve --livereload
```

## Structure

```
docs/
├── _config.yml           # Jekyll configuration
├── Gemfile              # Ruby dependencies
├── index.md             # Homepage
├── getting-started/     # Installation and setup
├── dataset/             # MERR dataset documentation
├── training/            # Training guides
├── evaluation/          # Evaluation benchmarks
├── demo/                # Demo usage
├── api/                 # API documentation (EN/ZH)
├── about/               # About, citation, license
└── assets/              # Images and custom JS/CSS
```

## Contributing to Documentation

### Adding a New Page

1. Create a new markdown file in the appropriate directory
2. Add YAML front matter:
   ```yaml
   ---
   layout: default
   title: Your Page Title
   parent: Parent Page (if any)
   nav_order: 1
   ---
   ```
3. Write your content in Markdown
4. Test locally before committing

### Navigation

Pages are automatically added to the navigation based on:
- `nav_order`: Position in the menu
- `parent`: Creates hierarchical navigation
- `has_children`: Indicates parent pages

### Styling

Just the Docs provides built-in styling:
- Callouts: `{: .note }`, `{: .warning }`, `{: .tip }`
- Buttons: `[Button Text](url){: .btn }`
- Labels: `{: .label }`, `{: .label .label-blue }`

See [Just the Docs documentation](https://just-the-docs.com/) for more.

## Deployment

The site is automatically built and deployed by GitHub Pages when changes are pushed to the `main` branch.

### GitHub Pages Settings

1. Go to repository Settings → Pages
2. Source: Deploy from branch
3. Branch: `main`
4. Folder: `/docs`

## Troubleshooting

### Build Errors

If the site fails to build:
1. Check `_config.yml` for syntax errors
2. Ensure all links are valid
3. Review GitHub Pages build log

### Local Build Issues

```bash
# Clean and rebuild
bundle exec jekyll clean
bundle exec jekyll build

# Update dependencies
bundle update
```

## Links

- **Live Site**: https://zebangcheng.github.io/Emotion-LLaMA/
- **Repository**: https://github.com/ZebangCheng/Emotion-LLaMA
- **Just the Docs**: https://just-the-docs.com/

## License

Documentation is licensed under CC BY-NC 4.0. See [License](about/license.md) for details.

