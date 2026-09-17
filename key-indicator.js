// key-indicator.js - on-screen arrow keys + space bar beside the puzzle grid.
//
// The game calls KeyIndicator.press(e.key) at the moment a keypress is
// actually acted on (i.e. when an event is recorded / the environment steps).
// That key lights up for a moment, then releases. Only one key is ever lit,
// and nothing is kept: this is a live "which button is being pressed right
// now" display, not a log.
//
// Mounts itself into #game-container on DOMContentLoaded and keeps itself
// positioned just to the right of #grid via a ResizeObserver (so it works
// for every grid size without per-game CSS).
(function () {
    const KEY_CLASS = {
        ArrowUp: 'kbd-up',
        ArrowDown: 'kbd-down',
        ArrowLeft: 'kbd-left',
        ArrowRight: 'kbd-right',
        ' ': 'kbd-space'
    };
    const HOLD_MS = 180;   // how long a key stays lit after a press

    let root = null;
    let lit = null;
    let releaseTimer = null;

    function build() {
        const container = document.getElementById('game-container');
        const grid = document.getElementById('grid');
        if (!container || !grid) return;

        root = document.createElement('div');
        root.id = 'key-indicator';
        root.setAttribute('aria-hidden', 'true');
        root.innerHTML =
            '<div class="kbd-key kbd-up">&#8593;</div>' +
            '<div class="kbd-key kbd-left">&#8592;</div>' +
            '<div class="kbd-key kbd-down">&#8595;</div>' +
            '<div class="kbd-key kbd-right">&#8594;</div>' +
            '<div class="kbd-key kbd-space"></div>';
        container.appendChild(root);

        // Sit just right of the grid: offset from the container's centre by
        // half the grid's rendered width (the grid is centred in it).
        const place = () => {
            const w = grid.getBoundingClientRect().width;
            root.style.setProperty('--kbd-grid-half', (w / 2) + 'px');
        };
        place();
        if (typeof ResizeObserver !== 'undefined') {
            new ResizeObserver(place).observe(grid);
        }
        window.addEventListener('resize', place);
    }

    function release() {
        if (lit) lit.classList.remove('active');
        lit = null;
        releaseTimer = null;
    }

    function press(key) {
        if (!root) return;
        const cls = KEY_CLASS[key];
        if (!cls) return;
        if (releaseTimer) clearTimeout(releaseTimer);
        release();
        lit = root.querySelector('.' + cls);
        if (!lit) return;
        lit.classList.add('active');
        releaseTimer = setTimeout(release, HOLD_MS);
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', build);
    } else {
        build();
    }

    window.KeyIndicator = { press: press };
})();
