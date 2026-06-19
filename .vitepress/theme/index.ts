// .vitepress/theme/index.ts
import DefaultTheme from 'vitepress/theme'
import { h } from 'vue'
import MyAdComponent from './MyAdComponent.vue'
import SidebarToggle from './SidebarToggle.vue'
import './custom.css'

export default {
  extends: DefaultTheme, // 继承 VitePress 的默认主题
  /* 如果后续有需要，你还可以在这里注册全局组件或做其他扩展 */
  Layout() {
    return h(DefaultTheme.Layout, null, {
      // 使用 aside-ads-after 插槽，将其注入到右侧栏广告位
      'aside-ads-after': () => h(MyAdComponent),

      // 2. 干净优雅地注入新封装的边栏开关组件
      'nav-bar-content-before': () => h(SidebarToggle)
    })
  }
}