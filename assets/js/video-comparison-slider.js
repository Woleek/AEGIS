// videoSlider.js

export function initVideoSlider() {
    const beforeLabel = document.querySelector('.video-label.before');
    const afterLabel = document.querySelector('.video-label.after');

    function updateLabels(percentage) {
        beforeLabel.style.opacity = percentage <= 3 ? "0" : "1";
        afterLabel.style.opacity = percentage >= 97 ? "0" : "1";
    }

    const slider = document.getElementById('video-slider');
    const afterWrapper = document.getElementById('video-after-wrapper');
    const handle = document.getElementById('slider-handle');

    let isDragging = false;

    slider.addEventListener('mousedown', startDrag);
    slider.addEventListener('touchstart', startDrag);

    window.addEventListener('mouseup', stopDrag);
    window.addEventListener('touchend', stopDrag);

    slider.addEventListener('mousemove', moveSlider);
    slider.addEventListener('touchmove', moveSlider);

    function startDrag() {
        isDragging = true;
        document.body.style.userSelect = 'none';
    }

    function stopDrag() {
        isDragging = false;
        document.body.style.userSelect = '';
    }

    function moveSlider(e) {
        if (!isDragging) return;

        e.preventDefault();

        const sliderLeft = slider.getBoundingClientRect().left;
        const clientX = e.clientX || e.touches?.[0]?.clientX;
        let pos = clientX - sliderLeft;

        const sliderWidth = slider.offsetWidth;
        pos = Math.max(0, Math.min(pos, sliderWidth));

        const percentage = (pos / sliderWidth) * 100;

        // afterWrapper is right-anchored (masked video on the right), so its
        // width is the region *right of* the handle, not left of it.
        afterWrapper.style.width = (100 - percentage) + '%';
        handle.style.left = percentage + '%';

        updateLabels(percentage);
    }

    return syncVideos();
}

// Keeps the "before" and "after" videos playing in lockstep so the
// comparison doesn't visibly drift apart over time or across loops, and
// exposes setAfterSource() so a variant selector can swap the "after" clip
// (e.g. different masking epsilons) without losing sync.
function syncVideos() {
    const master = document.querySelector('.video-before video');
    const follower = document.querySelector('.video-after video');

    if (!master || !follower) return { setAfterSource: () => {} };

    // We drive playback/looping manually instead of relying on the
    // `loop` attribute, since each <video> loops independently and
    // drifts further apart every cycle.
    master.loop = false;
    follower.loop = false;
    master.autoplay = false;
    follower.autoplay = false;

    const DRIFT_THRESHOLD = 0.08; // seconds
    let rafId = null;

    function whenReady(video) {
        if (video.readyState >= HTMLMediaElement.HAVE_FUTURE_DATA) {
            return Promise.resolve();
        }
        return new Promise((resolve) => {
            video.addEventListener('canplay', resolve, { once: true });
        });
    }

    async function playTogether(fromStart) {
        if (fromStart) {
            master.currentTime = 0;
            follower.currentTime = 0;
        }
        await Promise.all([whenReady(master), whenReady(follower)]);
        // Fire both play() calls back-to-back so their start times are
        // as close together as possible.
        await Promise.all([master.play(), follower.play()]);
    }

    function watchDrift() {
        if (!master.paused && !master.seeking) {
            const drift = follower.currentTime - master.currentTime;
            if (Math.abs(drift) > DRIFT_THRESHOLD) {
                follower.currentTime = master.currentTime;
            }
        }
        rafId = requestAnimationFrame(watchDrift);
    }

    // Restart both clips together whenever the master reaches the end,
    // rather than letting native looping desync them.
    master.addEventListener('ended', () => {
        playTogether(true);
    });

    // If the master stalls (buffering), pause the follower so it
    // doesn't run ahead; resume + resync once playback continues.
    master.addEventListener('waiting', () => {
        follower.pause();
    });
    master.addEventListener('playing', () => {
        follower.currentTime = master.currentTime;
        if (follower.paused) follower.play();
    });

    playTogether(true).then(() => {
        if (rafId === null) rafId = requestAnimationFrame(watchDrift);
    });

    window.addEventListener('beforeunload', () => {
        if (rafId) cancelAnimationFrame(rafId);
    });

    // Swap the "after" clip's source (e.g. a different masking epsilon) and
    // restart both videos together from 0 so they stay frame-locked.
    async function setAfterSource(url) {
        master.pause();
        follower.pause();
        follower.src = url;
        follower.load();
        await whenReady(follower);
        await playTogether(true);
    }

    // Swap both clips at once (e.g. a different subject changes both the
    // unmasked reference and the masked variant) and restart together.
    async function setSources({ before, after } = {}) {
        master.pause();
        follower.pause();
        if (before) {
            master.src = before;
            master.load();
        }
        if (after) {
            follower.src = after;
            follower.load();
        }
        await Promise.all([whenReady(master), whenReady(follower)]);
        await playTogether(true);
    }

    return { setAfterSource, setSources };
}
