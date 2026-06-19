<template>
  <button
    v-if="!isHome"
    class="sidebar-toggle-btn" 
    title="切换边栏" 
    @click="toggleSidebar"
  >
    <svg 
      xmlns="http://www.w3.org/2000/svg" 
      width="20" 
      height="20" 
      viewBox="0 0 24 24" 
      fill="none" 
      stroke="currentColor" 
      stroke-width="2" 
      stroke-linecap="round" 
      stroke-linejoin="round"
    >
      <line x1="3" y1="12" x2="21" y2="12"></line>
      <line x1="3" y1="6" x2="21" y2="6"></line>
      <line x1="3" y1="18" x2="21" y2="18"></line>
    </svg>
  </button>
</template>


<script setup lang="ts">
import { ref, computed } from 'vue'
import { useData } from 'vitepress'

const { page } = useData()
const isHome = computed(() => page.value.frontmatter.layout === 'home')

const isHidden = ref<boolean>(false)
const toggleSidebar = (event: MouseEvent): void => {
  event.stopPropagation()
  event.preventDefault()

  isHidden.value = !isHidden.value
  if (isHidden.value) {
    document.documentElement.classList.add('sidebar-hidden')
  } else {
    document.documentElement.classList.remove('sidebar-hidden')
  }
}
</script>

<style scoped>
.sidebar-toggle-btn {
  background: none;
  border: none;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 6px;
  margin-left: 12px;
  border-radius: 8px;
  color: var(--vp-c-text-1);
  pointer-events: auto;
}

.sidebar-toggle-btn:hover {
  background-color: var(--vp-c-default-soft);
}

@media (max-width: 959px) {
  .sidebar-toggle-btn {
    display: none;
  }
}
</style>

<style>
html.sidebar-hidden {
  --vp-sidebar-width: 0px !important;
}

html.sidebar-hidden .VPSidebar {
  display: none !important;
}

html.sidebar-hidden .VPNavBarTitle {
  display: none !important; 
}

html.sidebar-hidden .VPNavBar .container {
  max-width: 100% !important;
  padding: 0 !important;
}

html.sidebar-hidden .VPNavBar .divider {
  padding-left: 0 !important;
  margin-left: 0 !important;
  width: 100% !important;
}

</style>