/**
 * entity-cards.js — Horizontal scrollable entity card strip for per-answer sources.
 *
 * API:
 *   EntityCards.render(containerEl, sources) → { highlightCard(uri), unhighlightCard(uri), scrollToCard(uri) }
 */
const EntityCards = (() => {

    const FC_COLORS = {
        Thing:   '#4FC3F7',
        Actor:   '#FF8A65',
        Place:   '#81C784',
        Event:   '#BA68C8',
        Concept: '#FFD54F',
        Time:    '#F06292'
    };

    const FC_GRADIENTS = {
        Thing:   'linear-gradient(135deg, #4FC3F7 0%, #29B6F6 100%)',
        Actor:   'linear-gradient(135deg, #FF8A65 0%, #FF7043 100%)',
        Place:   'linear-gradient(135deg, #81C784 0%, #66BB6A 100%)',
        Event:   'linear-gradient(135deg, #BA68C8 0%, #AB47BC 100%)',
        Concept: 'linear-gradient(135deg, #FFD54F 0%, #FFC107 100%)',
        Time:    'linear-gradient(135deg, #F06292 0%, #EC407A 100%)'
    };

    const DEFAULT_GRADIENT = 'linear-gradient(135deg, #B0BEC5 0%, #90A4AE 100%)';

    /**
     * Get the best thumbnail URL from a source.
     */
    function getThumbnailUrl(source) {
        // Dataset images (array)
        if (source.images && source.images.length > 0) {
            return source.images[0].url;
        }
        // Wikidata image (object)
        if (source.image && source.image.thumbnail_url) {
            const url = source.image.thumbnail_url;
            if (typeof url === 'string' && url.startsWith('http')) return url;
        }
        return null;
    }

    /**
     * Get the full image URL for lightbox from a source.
     */
    function getFullImageUrl(source) {
        if (source.images && source.images.length > 0) {
            return source.images[0].url;
        }
        if (source.image && source.image.full_url) {
            return source.image.full_url;
        }
        return null;
    }

    /**
     * Render entity cards into containerEl.
     */
    function render(containerEl, sources) {
        if (!sources || sources.length === 0) {
            containerEl.style.display = 'none';
            return null;
        }

        containerEl.style.display = '';
        containerEl.innerHTML = '';

        const strip = document.createElement('div');
        strip.className = 'entity-cards-strip-inner';

        sources.forEach((source, idx) => {
            const fc = source.fc || '';
            const color = FC_COLORS[fc] || '#B0BEC5';
            const gradient = FC_GRADIENTS[fc] || DEFAULT_GRADIENT;
            const thumbUrl = getThumbnailUrl(source);
            const tripleCount = source.raw_triples ? source.raw_triples.length : 0;
            const label = source.entity_label || source.entity_uri.split('/').pop();
            const entityType = source.entity_type || 'unknown';

            const card = document.createElement('div');
            card.className = 'entity-card-v2';
            card.setAttribute('data-uri', source.entity_uri);
            card.setAttribute('data-source-index', idx);

            // Image area or FC gradient fallback
            const imageArea = document.createElement('div');
            imageArea.className = 'entity-card-v2-image';
            if (thumbUrl) {
                const img = document.createElement('img');
                img.src = thumbUrl;
                img.alt = label;
                img.loading = 'lazy';
                img.onerror = function() {
                    this.style.display = 'none';
                    imageArea.style.background = gradient;
                    // Add FC icon fallback
                    const icon = document.createElement('div');
                    icon.className = 'entity-card-v2-fc-icon';
                    icon.textContent = fc ? fc[0] : '?';
                    imageArea.appendChild(icon);
                };
                imageArea.appendChild(img);
            } else {
                imageArea.style.background = gradient;
                const icon = document.createElement('div');
                icon.className = 'entity-card-v2-fc-icon';
                icon.textContent = fc ? fc[0] : '?';
                imageArea.appendChild(icon);
            }
            card.appendChild(imageArea);

            // Info area
            const info = document.createElement('div');
            info.className = 'entity-card-v2-info';

            // FC dot + type
            const typeLine = document.createElement('div');
            typeLine.className = 'entity-card-v2-type';
            typeLine.innerHTML = `<span class="entity-card-v2-fc-dot" style="background-color: ${color}"></span>${entityType}`;
            info.appendChild(typeLine);

            // Label
            const labelEl = document.createElement('div');
            labelEl.className = 'entity-card-v2-label';
            labelEl.textContent = label;
            labelEl.title = label;
            info.appendChild(labelEl);

            // Triple count
            if (tripleCount > 0) {
                const tripleEl = document.createElement('div');
                tripleEl.className = 'entity-card-v2-triples';
                tripleEl.innerHTML = `<i class="fas fa-project-diagram"></i> ${tripleCount} triples`;
                info.appendChild(tripleEl);
            }

            card.appendChild(info);

            // Cross-linking events
            card.addEventListener('mouseenter', () => {
                containerEl.dispatchEvent(new CustomEvent('card-hover', {
                    bubbles: true,
                    detail: { uri: source.entity_uri, action: 'enter' }
                }));
            });
            card.addEventListener('mouseleave', () => {
                containerEl.dispatchEvent(new CustomEvent('card-hover', {
                    bubbles: true,
                    detail: { uri: source.entity_uri, action: 'leave' }
                }));
            });
            card.addEventListener('click', () => {
                containerEl.dispatchEvent(new CustomEvent('card-click', {
                    bubbles: true,
                    detail: { uri: source.entity_uri, source: source }
                }));
            });

            strip.appendChild(card);
        });

        containerEl.appendChild(strip);

        // Controller
        return {
            highlightCard(uri) {
                const card = strip.querySelector(`.entity-card-v2[data-uri="${CSS.escape(uri)}"]`);
                if (card) card.classList.add('entity-card-v2-highlight');
            },
            unhighlightCard(uri) {
                const card = strip.querySelector(`.entity-card-v2[data-uri="${CSS.escape(uri)}"]`);
                if (card) card.classList.remove('entity-card-v2-highlight');
            },
            scrollToCard(uri) {
                const card = strip.querySelector(`.entity-card-v2[data-uri="${CSS.escape(uri)}"]`);
                if (card) {
                    card.scrollIntoView({ behavior: 'smooth', block: 'nearest', inline: 'center' });
                    card.classList.add('entity-card-v2-highlight');
                    setTimeout(() => card.classList.remove('entity-card-v2-highlight'), 2000);
                }
            }
        };
    }

    return { render, getThumbnailUrl, getFullImageUrl, FC_COLORS };
})();
