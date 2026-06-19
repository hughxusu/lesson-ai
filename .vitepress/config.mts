import { withMermaid } from 'vitepress-plugin-mermaid'

// https://vitepress.dev/reference/site-config
export default withMermaid({
  title: "AI驱动的机器学习实战",
  description: "AI驱动机器学习实战",
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
        collapsed: true,
        items: [
          { text: '机器学习概述', link: '/docs/a-intro/index.md' }
        ]
      },
      {
        text: '机器学习经典算法',
        collapsed: true,
        items: [
          { text: 'K近邻', link: '/docs/b-base/01-knn.md' },
          { text: '线性回归', link: '/docs/b-base/02-线性.md' },
          { text: '梯度下降法', link: '/docs/b-base/03-梯度.md' },
          { text: '多项式回归与模型泛化', link: '/docs/b-base/04-多项式.md' },
          { text: '逻辑回归', link: '/docs/b-base/05-逻辑.md' },
          { text: '评价分类结果', link: '/docs/b-base/06-评价.md' },
          { text: '主成分分析', link: '/docs/b-base/07-pca.md' },
          { text: '支持向量机', link: '/docs/b-base/08-svm.md' },
          { text: '决策树', link: '/docs/b-base/09-决策树.md' },
          { text: '集成学习', link: '/docs/b-base/10-集成.md' },
        ]
      },
      {
        text: '深度学习',
        collapsed: true,
        items: [
          { text: '感知机', link: '/docs/c-dnn/01-感知机.md' },
          { text: '神经网络', link: '/docs/c-dnn/02-神经网络.md' },
          { text: '神经网络的学习', link: '/docs/c-dnn/03-网络学习.md' },
          { text: '反向传播法算法', link: '/docs/c-dnn/04-反向传播算法.md' },
          { text: '学习的相关技巧', link: '/docs/c-dnn/05-学习的相关技巧.md' },
          { text: '卷积神经网络', link: '/docs/c-dnn/06-卷积神经网络.md' },
          { text: '深度学习', link: '/docs/c-dnn/07-深度学习.md' },
        ]
      },
      {
        text: '机器视觉',
        collapsed: true,
        items: [
          { text: '简介', link: '/docs/d-cv/a-简介.md' },
          { text: '图像分类简介', link: '/docs/d-cv/b-分类-1.md' },
          { text: 'AlexNet', link: '/docs/d-cv/c-分类-2.md' },
          { text: 'VGGNet', link: '/docs/d-cv/d-分类-3.md' },
          { text: 'GoogLeNet', link: '/docs/d-cv/e-分类-4.md' },
          { text: '残差网络', link: '/docs/d-cv/f-分类-5.md' },
        ]
      },
      {
        text: '自然语言处理',
        collapsed: true,
        items: [
          { text: '简介', link: '/docs/e-nlp/01-简介.md' },
          { text: '文本预处理', link: '/docs/e-nlp/02-处理.md' },
          { text: 'fasttext工具', link: '/docs/e-nlp/03-fasttext.md' },
          { text: '简循环神经网络', link: '/docs/e-nlp/04-rnn.md' },
          { text: 'Transformer', link: '/docs/e-nlp/06-transformer.md' },
          { text: '预训练模型', link: '/docs/e-nlp/07-bert.md' },
          { text: 'Hugging Face', link: '/docs/e-nlp/08-huggingface.md' },
          { text: '模型微调入门', link: '/docs/e-nlp/09-微调入门.md' },
          { text: '模型量化', link: '/docs/e-nlp/10-模型量化.md' },
          { text: '大模型高效微调', link: '/docs/e-nlp/11-高效微调.md' },
          { text: '大模型微调实践', link: '/docs/e-nlp/12-llama.md' },
        ]
      },
      {
        text: '附录',
        collapsed: true,
        items: [
          { text: '自然语言处理', link: 'docs/z-others/05-nlp.md' }
        ]
      },
    ],

    outline: {
      label: '导航',
    },

    footer: {
      copyright: '徐夙 &copy; 2026 北方工业大学',
    },

    socialLinks: [
      { icon: 'github', link: 'https://github.com/hughxusu/lesson-ai' }
    ]
  }
})
