import { withMermaid } from 'vitepress-plugin-mermaid'

// https://vitepress.dev/reference/site-config
export default withMermaid({
  title: "机器学习",
  description: "机器学习",
  ignoreDeadLinks: true,
  base: '/lesson-ai/',
  markdown: {
    math: true,
  },
  head: [
    ['link', { rel: 'icon', href: '/lesson-ai/logo_icon.jpeg' }],
  ],
  themeConfig: {
    sidebar: [
      {
        text: '绪论',
        items: [
          { text: '机器学习介绍', link: '/docs/a-intro/index.md' }
        ]
      }
    ],

    outline: {
      label: '导航',
    },

    footer: {
      copyright: '徐夙 &copy; 2026 北方工业大学',
    },

    socialLinks: [
      { icon: 'github', link: 'https://github.com/vuejs/vitepress' }
    ]
  }
})
