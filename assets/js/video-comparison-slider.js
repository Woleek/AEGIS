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

        afterWrapper.style.width = percentage + '%';
        handle.style.left = percentage + '%';

        updateLabels(percentage);
    }
}
