import type { BaseLayoutProps } from 'fumadocs-ui/layouts/shared';

export function baseOptions(): BaseLayoutProps {
  return {
    nav: { title: 'MolFrame', url: '/docs' },
    githubUrl: 'https://github.com/molstation/molframe',
    links: [{ text: 'MolGFX', url: 'https://molstation.github.io/molgfx/', type: 'main' }],
  };
}