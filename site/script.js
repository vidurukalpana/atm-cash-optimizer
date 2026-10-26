// Mobile menu
const toggle = document.querySelector('.menu-toggle');
const links = document.querySelector('.nav-links');
toggle.addEventListener('click', () => {
  const open = links.classList.toggle('open');
  toggle.setAttribute('aria-expanded', open);
});
links.querySelectorAll('a').forEach((a) => a.addEventListener('click', () => links.classList.remove('open')));

// Screenshot not added yet: show a neat placeholder instead of a broken image
document.querySelectorAll('img[data-label]').forEach((img) => {
  img.addEventListener('error', () => {
    const box = document.createElement('div');
    box.className = 'missing';
    box.textContent = img.dataset.label;
    img.replaceWith(box);
  });
});

// Click a screenshot to enlarge it
const lightbox = document.querySelector('.lightbox');
const large = lightbox.querySelector('img');
document.addEventListener('click', (event) => {
  const img = event.target.closest('.gallery img, .hero-shot img, .chart img');
  if (img) {
    large.src = img.src;
    large.alt = img.alt;
    lightbox.hidden = false;
  } else if (event.target === lightbox || event.target.closest('.lightbox-close')) {
    lightbox.hidden = true;
  }
});
document.addEventListener('keydown', (event) => {
  if (event.key === 'Escape') lightbox.hidden = true;
});
