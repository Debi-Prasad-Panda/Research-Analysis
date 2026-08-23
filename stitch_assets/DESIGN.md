---
name: Synthetica Research Atlas
colors:
  surface: '#131314'
  surface-dim: '#131314'
  surface-bright: '#3a393a'
  surface-container-lowest: '#0e0e0f'
  surface-container-low: '#1c1b1c'
  surface-container: '#201f20'
  surface-container-high: '#2a2a2a'
  surface-container-highest: '#353435'
  on-surface: '#e5e2e2'
  on-surface-variant: '#c7c6cc'
  inverse-surface: '#e5e2e2'
  inverse-on-surface: '#313031'
  outline: '#919096'
  outline-variant: '#46464b'
  surface-tint: '#c5c5d2'
  primary: '#c5c5d2'
  on-primary: '#2e303a'
  primary-container: '#0f111a'
  on-primary-container: '#7b7c88'
  inverse-primary: '#5c5e69'
  secondary: '#bcc7de'
  on-secondary: '#263143'
  secondary-container: '#3e495d'
  on-secondary-container: '#aeb9d0'
  tertiary: '#d5c4b0'
  on-tertiary: '#392f21'
  tertiary-container: '#181005'
  on-tertiary-container: '#897b69'
  error: '#ffb4ab'
  on-error: '#690005'
  error-container: '#93000a'
  on-error-container: '#ffdad6'
  primary-fixed: '#e1e1ef'
  primary-fixed-dim: '#c5c5d2'
  on-primary-fixed: '#191b24'
  on-primary-fixed-variant: '#454651'
  secondary-fixed: '#d8e3fb'
  secondary-fixed-dim: '#bcc7de'
  on-secondary-fixed: '#111c2d'
  on-secondary-fixed-variant: '#3c475a'
  tertiary-fixed: '#f2e0cb'
  tertiary-fixed-dim: '#d5c4b0'
  on-tertiary-fixed: '#231a0d'
  on-tertiary-fixed-variant: '#504536'
  background: '#131314'
  on-background: '#e5e2e2'
  surface-variant: '#353435'
typography:
  display-lg:
    fontFamily: Inter
    fontSize: 48px
    fontWeight: '700'
    lineHeight: '1.1'
    letterSpacing: -0.02em
  headline-md:
    fontFamily: Inter
    fontSize: 24px
    fontWeight: '600'
    lineHeight: '1.3'
  body-base:
    fontFamily: Inter
    fontSize: 16px
    fontWeight: '400'
    lineHeight: '1.6'
  formula-md:
    fontFamily: JetBrains Mono
    fontSize: 15px
    fontWeight: '450'
    lineHeight: '1.5'
  label-caps:
    fontFamily: JetBrains Mono
    fontSize: 12px
    fontWeight: '600'
    lineHeight: '1'
    letterSpacing: 0.08em
  headline-md-mobile:
    fontFamily: Inter
    fontSize: 20px
    fontWeight: '600'
    lineHeight: '1.3'
rounded:
  sm: 0.125rem
  DEFAULT: 0.25rem
  md: 0.375rem
  lg: 0.5rem
  xl: 0.75rem
  full: 9999px
spacing:
  unit: 4px
  gutter: 24px
  margin: 40px
  container-max: 1440px
---

## Brand & Style

The design system is engineered for high-density academic exploration and data synthesis. It targets researchers, AI engineers, and doctoral students who require a "Command Center" environment that balances the rigor of traditional academic publishing with the velocity of cutting-edge technology.

The aesthetic follows a **Futuristic Academic** style—a hybrid of **Glassmorphism** and **Minimalism**. It utilizes deep, atmospheric layering to create a sense of infinite digital space. The interface should feel "high-fidelity," utilizing subtle glows and micro-interactions to signal active processing and data-rich environments. The emotional response is one of focused immersion, intellectual authority, and technical precision.

## Colors

The palette is centered on **Midnight Indigo** (`#0F111A`) and **Slate** (`#1E293B`) to reduce eye strain during prolonged research sessions. 

- **Primary Background**: Deep obsidian, providing a void-like canvas for glass layers.
- **Cyber Lime**: Used exclusively for high-priority highlights, active states, and calls to action. It represents "insight" or "discovery."
- **Mathematical Blue**: Applied to formulas, technical symbols, and secondary data visualizations to differentiate theoretical content from interface controls.
- **Surface Treatment**: Surfaces use semi-transparent variants of the neutral palette to support the glassmorphism effect, typically at 60-80% opacity with a 20px backdrop blur.

## Typography

The typographic system prioritizes legibility in high-density data views. 

- **Primary Interface**: **Inter** is used for all UI elements and body copy. It provides a clean, neutral tone that does not distract from the research content.
- **Technical Content**: **JetBrains Mono** is utilized for mathematical formulas, code blocks, and metadata labels. This monospaced font provides the necessary character distinction for academic precision.
- **Hierarchy**: Use `label-caps` for all-caps metadata (e.g., DOI numbers, citations). Large headlines should use tighter letter-spacing to maintain a "scientific journal" density.

## Layout & Spacing

This design system employs a **Fluid Grid** model with a strict 4px baseline rhythm. 

- **Desktop (1440px+)**: A 12-column grid with generous 40px external margins. Sidebars for navigation and "Search Insights" are docked or collapsible, using a 280px fixed width.
- **Tablet (768px - 1439px)**: Transitions to an 8-column grid. Gutters reduce to 16px.
- **Mobile (< 767px)**: A 4-column grid. Heavy use of vertical stacking. The "Command Center" feel is maintained via a persistent bottom-docked utility bar.
- **Content Density**: Maintain "Comfortable" spacing for reading abstracts, but "Compact" spacing for data tables and bibliography lists.

## Elevation & Depth

Visual hierarchy is achieved through **Tonal Layers** and **Glassmorphism** rather than traditional drop shadows.

- **Level 0 (Base)**: The Midnight Indigo background.
- **Level 1 (Panels)**: Semi-transparent Slate backgrounds (`rgba(30, 41, 59, 0.7)`) with a 16px backdrop blur.
- **Level 2 (Modals/Popovers)**: Higher transparency Slate with a subtle 1px inner border of `rgba(255, 255, 255, 0.1)` to simulate a glass edge.
- **Active State Highlights**: Elements in focus or "processing" should feature a subtle outer glow (4px blur) using the Cyber Lime color at 30% opacity.

## Shapes

The shape language is **Soft** but precise. 

- **Standard Radius**: 4px (0.25rem) for most buttons, inputs, and small containers, maintaining a technical, "engineered" look.
- **Large Radius**: 12px (0.75rem) for main content cards and glass panels to soften the overall interface and make the "frosted glass" look more organic.
- **Interactive Elements**: Hover states should involve a transition from the standard radius to a slightly more rounded profile (transitioning from `rounded` to `rounded-lg`) to signal interactivity.

## Components

- **Primary Buttons**: Solid Cyber Lime background with black text for maximum contrast. No shadows; use a 1px border glow on hover.
- **Input Fields**: Ghost-style with a semi-transparent Slate fill and a 1px Slate-400 border. Upon focus, the border transitions to Mathematical Blue.
- **Research Cards**: Glassmorphic panels with `label-caps` for category tags. On hover, the border-color should shift from Slate to Cyber Lime.
- **Formula Blocks**: Integrated containers using the Mathematical Blue color for text, set against a slightly darker, opaque version of the background to ensure LaTeX or MathML clarity.
- **Progress Indicators**: Thin, glowing Cyber Lime lines that appear at the top of panels or the viewport during "Deep Research" synthesis phases.
- **Citations**: Inline "Chips" using `JetBrains Mono`, styled with a subtle blue tint to differentiate them from standard body text.
