document.addEventListener('DOMContentLoaded', function() {
    const chatContainer = document.getElementById('chat-container');
    const questionInput = document.getElementById('question-input');
    const sendButton = document.getElementById('send-button');
    let exampleQuestions = document.querySelectorAll('.example-question');

    // Dataset management state
    let currentDatasetId = null;
    // Conversation history for follow-up context
    let chatHistory = [];
    // Configurable prompt templates (loaded from prompts.yaml via API)
    let entityCardPrompt = 'Tell me about {label}';
    let entityDetailPrompt = 'Tell me more about {label}';
    const datasetSelect = document.getElementById('dataset-select');
    const datasetStatus = document.getElementById('dataset-status');

    // Track active mini-graph controllers for cleanup
    let activeMiniGraphs = [];

    // Message ID counter for scoping
    let messageIdCounter = 0;

    // ========== Dataset Management ==========

    async function loadDatasets() {
        try {
            const response = await fetch('/api/datasets');
            if (response.ok) {
                const data = await response.json();
                datasetSelect.innerHTML = '<option value="" disabled selected>Select Dataset...</option>';
                data.datasets.forEach(ds => {
                    const option = document.createElement('option');
                    option.value = ds.id;
                    let displayText = ds.display_name;
                    if (!ds.has_cache) displayText += ' (not built)';
                    else if (!ds.initialized) displayText += ' (not loaded)';
                    option.textContent = displayText;
                    datasetSelect.appendChild(option);
                });
                if (data.default) {
                    datasetSelect.value = data.default;
                    await switchDataset(data.default);
                }
            }
        } catch (error) {
            console.error('Error loading datasets:', error);
            datasetStatus.textContent = 'Error';
            datasetStatus.className = 'ms-2 badge bg-danger';
        }
    }

    async function switchDataset(datasetId) {
        if (!datasetId) return;
        datasetStatus.textContent = 'Loading...';
        datasetStatus.className = 'ms-2 badge bg-warning';
        sendButton.disabled = true;

        try {
            const response = await fetch(`/api/datasets/${datasetId}/select`, { method: 'POST' });
            if (response.ok) {
                const data = await response.json();
                currentDatasetId = datasetId;
                updateInterface(data.interface);
                loadTopEntities(datasetId);
                clearChat();
                addWelcomeMessage(data.interface.welcome_message);
                datasetStatus.textContent = 'Active';
                datasetStatus.className = 'ms-2 badge bg-success';
            } else {
                const errorData = await response.json();
                console.error('Error selecting dataset:', errorData.error);
                datasetStatus.textContent = 'Error';
                datasetStatus.className = 'ms-2 badge bg-danger';
            }
        } catch (error) {
            console.error('Error switching dataset:', error);
            datasetStatus.textContent = 'Error';
            datasetStatus.className = 'ms-2 badge bg-danger';
        } finally {
            sendButton.disabled = false;
        }
    }

    function updateInterface(config) {
        if (config.page_title) document.title = config.page_title;
        const headerTitle = document.getElementById('header-title');
        if (headerTitle && config.header_title) {
            headerTitle.innerHTML = `<i class="fas fa-comments me-2"></i>${config.header_title}`;
        }
        if (config.input_placeholder) questionInput.placeholder = config.input_placeholder;
        updateExampleQuestions(config.example_questions);
        // Load configurable prompt templates
        if (config.prompts) {
            if (config.prompts.entity_card_prompt) entityCardPrompt = config.prompts.entity_card_prompt;
            if (config.prompts.entity_detail_prompt) entityDetailPrompt = config.prompts.entity_detail_prompt;
        }
    }

    function updateExampleQuestions(questions) {
        if (!questions || questions.length === 0) return;
        const container = document.getElementById('example-questions-row');
        if (!container) return;

        const leftQuestions = questions.slice(0, 2);
        const rightQuestions = questions.slice(2);

        let html = '<div class="col-md-6"><ul>';
        leftQuestions.forEach(q => {
            html += `<li><a href="#" class="example-question">${q}</a></li>`;
        });
        html += '</ul></div><div class="col-md-6"><ul>';
        rightQuestions.forEach(q => {
            html += `<li><a href="#" class="example-question">${q}</a></li>`;
        });
        html += '</ul></div>';
        container.innerHTML = html;

        exampleQuestions = document.querySelectorAll('.example-question');
        exampleQuestions.forEach(question => {
            question.addEventListener('click', (e) => {
                e.preventDefault();
                questionInput.value = question.textContent;
                sendQuestion(question.textContent);
            });
        });
    }

    function clearChat() {
        // Dispose any active mini-graphs
        activeMiniGraphs.forEach(ctrl => { if (ctrl) ctrl.dispose(); });
        activeMiniGraphs = [];
        chatContainer.innerHTML = '';
        // Reset exploration panel to expanded for new dataset discovery
        setExplorationCollapsed(false);
    }

    function addWelcomeMessage(message) {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message assistant-message';
        messageDiv.innerHTML = `<div class="answer-text"><p>${message}</p></div>`;
        chatContainer.appendChild(messageDiv);
    }

    datasetSelect.addEventListener('change', (e) => switchDataset(e.target.value));
    loadDatasets();

    async function loadSystemInfo() {
        try {
            const response = await fetch('/api/info');
            if (response.ok) {
                const data = await response.json();
                if (!currentDatasetId) {
                    const initialMessage = chatContainer.querySelector('.assistant-message p');
                    if (initialMessage && data.dataset_description) {
                        initialMessage.textContent = `Hello! I'm a chatbot assistant. I can answer questions about ${data.dataset_description}. How can I help you today?`;
                    }
                }
            }
        } catch (error) {
            console.error('Error loading system info:', error);
        }
    }
    loadSystemInfo();

    // ========== Entity Detail Panel ==========

    const detailCard = document.getElementById('entity-detail-card');
    const detailBody = document.getElementById('entity-detail-body');
    const detailTitle = document.getElementById('entity-detail-title');
    const detailCloseBtn = document.getElementById('entity-detail-close');

    function showEntityDetail(source) {
        if (!detailCard || !detailBody) return;

        const rightColumn = document.getElementById('right-column');
        if (rightColumn) rightColumn.style.display = 'block';
        detailCard.style.display = 'block';

        const fc = source.fc || '';
        const fcColor = FC_COLORS[fc] || '#B0BEC5';
        const label = source.entity_label || source.entity_uri.split('/').pop();
        const entityType = source.entity_type || 'unknown';
        const pagerank = source.pagerank || 0;

        // Update header title
        detailTitle.textContent = label;

        let html = '';

        // 1. Image or FC gradient hero
        const thumbUrl = EntityCards.getThumbnailUrl(source);
        const fullUrl = EntityCards.getFullImageUrl(source) || thumbUrl;
        const gradient = EntityCards.FC_COLORS ? '' : '';
        const FC_GRADIENTS = {
            Thing:   'linear-gradient(135deg, #4FC3F7 0%, #29B6F6 100%)',
            Actor:   'linear-gradient(135deg, #FF8A65 0%, #FF7043 100%)',
            Place:   'linear-gradient(135deg, #81C784 0%, #66BB6A 100%)',
            Event:   'linear-gradient(135deg, #BA68C8 0%, #AB47BC 100%)',
            Concept: 'linear-gradient(135deg, #FFD54F 0%, #FFC107 100%)',
            Time:    'linear-gradient(135deg, #F06292 0%, #EC407A 100%)'
        };
        const bgGradient = FC_GRADIENTS[fc] || 'linear-gradient(135deg, #B0BEC5 0%, #90A4AE 100%)';

        if (thumbUrl) {
            html += `<div class="entity-detail-image" onclick="openDetailLightbox('${fullUrl}', '${label.replace(/'/g, "\\'")}')">
                <img src="${thumbUrl}" alt="${label}" onerror="this.style.display='none'; this.parentElement.style.background='${bgGradient}'; this.parentElement.innerHTML+='<div class=\\'detail-fc-icon\\'>${fc ? fc[0] : '?'}</div>'">
            </div>`;
        } else {
            html += `<div class="entity-detail-image" style="background: ${bgGradient}">
                <div class="detail-fc-icon">${fc ? fc[0] : '?'}</div>
            </div>`;
        }

        // 2. Metadata bar
        html += `<div class="entity-detail-meta">
            <span class="detail-fc-badge" style="background-color: ${fcColor}">${fc || 'Unknown'}</span>
            <span class="detail-type-label">${entityType}</span>
            ${pagerank > 0 ? `<span class="detail-pagerank" title="PageRank score"><i class="fas fa-chart-line me-1"></i>${pagerank.toFixed(6)}</span>` : ''}
        </div>`;

        // 3. Group triples by edge_type and predicate
        const triples = source.raw_triples || [];
        const entityUri = source.entity_uri;

        const frTriples = triples.filter(t => t.edge_type === 'fr');
        const rdfTriples = triples.filter(t => t.edge_type !== 'fr');

        // Build relationship section
        function buildRelSection(sectionTriples, title, iconClass, sectionId) {
            if (sectionTriples.length === 0) return '';

            // Group by predicate_label, separating outgoing vs incoming
            const outgoing = {};
            const incoming = {};
            sectionTriples.forEach(t => {
                const pred = t.predicate_label || t.predicate.split('/').pop().split('#').pop();
                if (t.subject === entityUri) {
                    if (!outgoing[pred]) outgoing[pred] = [];
                    outgoing[pred].push({ label: t.object_label || t.object.split('/').pop(), uri: t.object });
                } else {
                    if (!incoming[pred]) incoming[pred] = [];
                    incoming[pred].push({ label: t.subject_label || t.subject.split('/').pop(), uri: t.subject });
                }
            });

            let s = `<div class="detail-section">
                <div class="detail-section-header" onclick="toggleDetailSection('${sectionId}')">
                    <i class="fas fa-caret-right" id="${sectionId}-icon"></i>
                    <i class="${iconClass}"></i> ${title}
                    <span class="detail-section-count">${sectionTriples.length}</span>
                </div>
                <div class="detail-section-body" id="${sectionId}">`;

            const MAX_TARGETS = 10;

            function renderTargets(targets) {
                let out = '';
                const shown = targets.slice(0, MAX_TARGETS);
                shown.forEach(t => {
                    out += `<div class="detail-rel-target">
                        <span class="detail-rel-target-label">${t.label}</span>
                    </div>`;
                });
                if (targets.length > MAX_TARGETS) {
                    out += `<div class="detail-rel-more">and ${targets.length - MAX_TARGETS} more</div>`;
                }
                return out;
            }

            // Outgoing relationships
            for (const [pred, targets] of Object.entries(outgoing)) {
                s += `<div class="detail-rel-group">
                    <div class="detail-rel-predicate">
                        <span class="direction-icon"><i class="fas fa-arrow-right"></i></span> ${pred}
                        <span class="detail-rel-count">${targets.length}</span>
                    </div>`;
                s += renderTargets(targets);
                s += '</div>';
            }

            // Incoming relationships
            for (const [pred, targets] of Object.entries(incoming)) {
                s += `<div class="detail-rel-group">
                    <div class="detail-rel-predicate">
                        <span class="direction-icon"><i class="fas fa-arrow-left"></i></span> ${pred}
                        <span class="detail-rel-count">${targets.length}</span>
                    </div>`;
                s += renderTargets(targets);
                s += '</div>';
            }

            s += '</div></div>';
            return s;
        }

        html += buildRelSection(frTriples, 'Fundamental Relationships', 'fas fa-project-diagram', 'detail-fr');
        html += buildRelSection(rdfTriples, 'RDF Properties', 'fas fa-code-branch', 'detail-rdf');

        // 4. Entity document text
        if (source.doc_text) {
            html += `<details class="detail-doc-section">
                <summary><i class="fas fa-file-alt me-1"></i>Entity Document</summary>
                <div class="detail-doc-content">${marked.parse(source.doc_text)}</div>
            </details>`;
        }

        // 5. Wikidata link
        if (source.wikidata_url) {
            html += `<div class="detail-wikidata-link">
                <a href="${source.wikidata_url}" target="_blank" class="btn btn-sm btn-outline-success">
                    <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/f/ff/Wikidata-logo.svg/20px-Wikidata-logo.svg.png" alt="Wikidata" style="height:14px; margin-right:4px; vertical-align:middle;">
                    View on Wikidata
                </a>
            </div>`;
        }

        // 6. Ask about this entity button
        html += `<div class="detail-ask-button">
            <button class="btn btn-sm btn-primary" id="entity-detail-ask-btn"
                    data-label="${label.replace(/"/g, '&quot;')}">
                <i class="fas fa-comment me-1"></i>Ask about this entity
            </button>
        </div>`;

        detailBody.innerHTML = html;

        // Wire ask button
        const askBtn = document.getElementById('entity-detail-ask-btn');
        if (askBtn) {
            askBtn.addEventListener('click', function() {
                const entityLabel = this.getAttribute('data-label');
                const question = entityDetailPrompt.replace('{label}', entityLabel);
                questionInput.value = question;
                sendQuestion(question);
            });
        }

        // Auto-expand FR section if present
        if (frTriples.length > 0) {
            toggleDetailSection('detail-fr');
        } else if (rdfTriples.length > 0) {
            toggleDetailSection('detail-rdf');
        }

        // Scroll detail card into view
        detailCard.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    // Toggle a detail section open/closed
    window.toggleDetailSection = function(sectionId) {
        const body = document.getElementById(sectionId);
        const icon = document.getElementById(sectionId + '-icon');
        if (!body) return;
        body.classList.toggle('expanded');
        if (icon) icon.classList.toggle('expanded');
    };

    // Lightbox for detail panel images
    window.openDetailLightbox = function(fullUrl, label) {
        if (lightbox && lightboxImage) {
            lightboxImage.src = fullUrl;
            lightboxLabel.textContent = label;
            lightboxCommonsLink.href = fullUrl;
            const linkText = lightboxCommonsLink.querySelector('.lightbox-link-text');
            if (linkText) linkText.textContent = 'View image';
            lightbox.style.display = 'flex';
            document.body.style.overflow = 'hidden';
        }
    };

    function hideEntityDetail() {
        if (detailCard) detailCard.style.display = 'none';
    }

    if (detailCloseBtn) {
        detailCloseBtn.addEventListener('click', hideEntityDetail);
    }

    // ========== FC Colors & Entity Exploration Panel ==========

    const FC_COLORS = {
        Thing: '#4FC3F7', Actor: '#FF8A65', Place: '#81C784',
        Event: '#BA68C8', Concept: '#FFD54F', Time: '#F06292'
    };
    const FC_ORDER = ['Thing', 'Actor', 'Place', 'Event', 'Concept', 'Time'];

    // Exploration panel collapse state
    let explorationCollapsed = false;
    const explorationToggle = document.getElementById('exploration-toggle');
    const explorationToggleIcon = document.getElementById('exploration-toggle-icon');
    const explorationWrapper = document.getElementById('entity-cards-wrapper');
    const explorationFcSummary = document.getElementById('exploration-fc-summary');

    function setExplorationCollapsed(collapsed) {
        explorationCollapsed = collapsed;
        if (!explorationWrapper || !explorationToggleIcon) return;

        if (collapsed) {
            explorationWrapper.classList.add('collapsed');
            explorationToggleIcon.classList.add('collapsed');
            if (explorationFcSummary && explorationFcSummary.innerHTML) {
                explorationFcSummary.style.display = 'flex';
            }
        } else {
            explorationWrapper.classList.remove('collapsed');
            explorationToggleIcon.classList.remove('collapsed');
            if (explorationFcSummary) explorationFcSummary.style.display = 'none';
        }
    }

    if (explorationToggle) {
        explorationToggle.addEventListener('click', () => {
            setExplorationCollapsed(!explorationCollapsed);
        });
    }

    /** Build the compact FC summary chips shown when collapsed */
    function buildFcSummary(groups) {
        if (!explorationFcSummary) return;
        let html = '';
        FC_ORDER.forEach(fc => {
            const items = groups[fc];
            if (!items || items.length === 0) return;
            const color = FC_COLORS[fc] || '#999';
            html += `<span class="fc-summary-chip">
                <span class="fc-dot" style="background-color: ${color}"></span>
                ${fc} <strong>${items.length}</strong>
            </span>`;
        });
        explorationFcSummary.innerHTML = html;
    }

    async function loadTopEntities(datasetId) {
        const rightColumn = document.getElementById('right-column');
        const explorationCard = document.getElementById('entity-exploration-card');
        const container = document.getElementById('entity-cards-container');
        const countBadge = document.getElementById('entity-count-badge');
        if (!rightColumn || !explorationCard || !container) return;

        try {
            const response = await fetch(`/api/datasets/${datasetId}/top-entities?top_n=30`);
            if (!response.ok) return;

            const data = await response.json();
            const entities = data.entities || [];
            if (entities.length === 0) return;

            rightColumn.style.display = 'block';
            explorationCard.style.display = 'block';
            countBadge.textContent = entities.length;

            // Start expanded for discovery
            setExplorationCollapsed(false);

            const groups = {};
            entities.forEach(entity => {
                const fc = entity.fc || 'Thing';
                if (!groups[fc]) groups[fc] = [];
                groups[fc].push(entity);
            });

            // Build FC summary chips for collapsed state
            buildFcSummary(groups);

            let html = '';
            FC_ORDER.forEach(fc => {
                const items = groups[fc];
                if (!items || items.length === 0) return;
                const color = FC_COLORS[fc] || '#999';

                html += `<div class="fc-group">`;
                html += `<div class="fc-group-header">
                    <span class="fc-dot" style="background-color: ${color}"></span>
                    <span>${fc}</span>
                    <span class="fc-group-count">${items.length}</span>
                </div>`;

                items.forEach((entity, idx) => {
                    const label = entity.label || entity.uri;
                    html += `<div class="entity-card" style="border-left-color: ${color}"
                                  data-label="${label.replace(/"/g, '&quot;')}"
                                  title="${label}">
                        <div class="entity-card-body">
                            <div class="entity-card-label">${label}</div>
                        </div>
                        <span class="entity-card-rank">#${idx + 1}</span>
                    </div>`;
                });
                html += `</div>`;
            });

            container.innerHTML = html;

            container.querySelectorAll('.entity-card').forEach(card => {
                card.addEventListener('click', function() {
                    const label = this.getAttribute('data-label');
                    const question = entityCardPrompt.replace('{label}', label);
                    questionInput.value = question;
                    sendQuestion(question);
                });
            });
        } catch (error) {
            console.error('Error loading top entities:', error);
        }
    }

    // ========== Image Gallery ==========

    function updateImageGallery(sources) {
        const galleryContainer = document.getElementById('image-gallery');
        const galleryCard = document.getElementById('image-gallery-card');
        const rightColumn = document.getElementById('right-column');
        if (!galleryContainer || !galleryCard) return;

        const imageMap = new Map();
        sources.forEach(source => {
            if (source.image && source.image.thumbnail_url) {
                const thumbnailUrl = source.image.thumbnail_url;
                if (typeof thumbnailUrl === 'string' && thumbnailUrl.startsWith('http') &&
                    !thumbnailUrl.includes('[') && !thumbnailUrl.includes(']')) {
                    const key = thumbnailUrl;
                    if (!imageMap.has(key)) {
                        imageMap.set(key, {
                            thumbnailUrl: thumbnailUrl,
                            fullUrl: source.image.full_url || thumbnailUrl,
                            linkUrl: source.image.url || thumbnailUrl,
                            label: source.entity_label || 'Image',
                            source: source.image.source || 'wikidata',
                            entityUri: source.entity_uri,
                            fc: source.fc || ''
                        });
                    }
                }
            }
            if (source.images && Array.isArray(source.images)) {
                source.images.forEach(img => {
                    const imgUrl = img.url;
                    if (typeof imgUrl === 'string' && imgUrl.startsWith('http') && !imageMap.has(imgUrl)) {
                        imageMap.set(imgUrl, {
                            thumbnailUrl: imgUrl,
                            fullUrl: imgUrl,
                            linkUrl: imgUrl,
                            label: source.entity_label || 'Image',
                            source: img.source || 'dataset',
                            entityUri: source.entity_uri,
                            fc: source.fc || ''
                        });
                    }
                });
            }
        });
        const images = Array.from(imageMap.values());

        if (images.length === 0) {
            galleryCard.style.display = 'none';
            galleryContainer.innerHTML = '';
            return;
        }

        if (rightColumn) rightColumn.style.display = 'block';
        galleryCard.style.display = 'block';

        let galleryHtml = '<div class="image-gallery-grid" id="image-gallery-grid">';
        images.forEach((img, index) => {
            const fcColor = FC_COLORS[img.fc] || '#B0BEC5';
            galleryHtml += `
                <div class="gallery-item" data-image-index="${index}" data-entity-uri="${img.entityUri}">
                    <a href="#" class="gallery-image-link"
                       data-full-url="${img.fullUrl}"
                       data-commons-url="${img.linkUrl}"
                       data-label="${img.label}"
                       data-source="${img.source}"
                       data-entity-uri="${img.entityUri}"
                       onclick="openLightbox(event, this)">
                        <img src="${img.thumbnailUrl}"
                             alt="${img.label}"
                             class="gallery-thumbnail"
                             loading="lazy"
                             onerror="handleImageError(this)">
                        <div class="gallery-overlay">
                            <span class="gallery-label">${img.label}</span>
                            <span class="gallery-action"><i class="fas fa-search-plus"></i> View larger</span>
                        </div>
                        <div class="gallery-fc-border" style="background-color: ${fcColor}"></div>
                    </a>
                </div>
            `;
        });
        galleryHtml += '</div>';
        galleryHtml += `<div class="gallery-count text-muted small mt-2" id="gallery-count">${images.length} image${images.length > 1 ? 's' : ''} found</div>`;
        galleryContainer.innerHTML = galleryHtml;

        // Cross-link: click gallery image → highlight entity card + mini-graph node
        galleryContainer.querySelectorAll('.gallery-item').forEach(item => {
            item.addEventListener('click', function(e) {
                // Don't interfere with lightbox
                if (e.target.closest('.gallery-image-link')) return;
            });
            item.addEventListener('mouseenter', function() {
                const uri = this.getAttribute('data-entity-uri');
                document.dispatchEvent(new CustomEvent('gallery-hover', {
                    detail: { uri, action: 'enter' }
                }));
            });
            item.addEventListener('mouseleave', function() {
                const uri = this.getAttribute('data-entity-uri');
                document.dispatchEvent(new CustomEvent('gallery-hover', {
                    detail: { uri, action: 'leave' }
                }));
            });
        });
    }

    // Gallery highlight functions
    window.highlightGalleryImage = function(uri) {
        document.querySelectorAll(`.gallery-item[data-entity-uri="${CSS.escape(uri)}"]`).forEach(item => {
            item.classList.add('gallery-item-highlight');
        });
    };

    window.unhighlightGalleryImage = function(uri) {
        document.querySelectorAll(`.gallery-item[data-entity-uri="${CSS.escape(uri)}"]`).forEach(item => {
            item.classList.remove('gallery-item-highlight');
        });
    };

    window.handleImageError = function(imgElement) {
        const galleryItem = imgElement.closest('.gallery-item');
        if (galleryItem) {
            galleryItem.style.display = 'none';
            const galleryGrid = document.getElementById('image-gallery-grid');
            const galleryCount = document.getElementById('gallery-count');
            const galleryCard = document.getElementById('image-gallery-card');
            if (galleryGrid) {
                const visibleItems = galleryGrid.querySelectorAll('.gallery-item:not([style*="display: none"])');
                const count = visibleItems.length;
                if (count === 0) {
                    if (galleryCard) galleryCard.style.display = 'none';
                } else if (galleryCount) {
                    galleryCount.textContent = `${count} image${count > 1 ? 's' : ''} found`;
                }
            }
        }
    };

    function clearImageGallery() {
        const galleryContainer = document.getElementById('image-gallery');
        const galleryCard = document.getElementById('image-gallery-card');
        if (galleryContainer) galleryContainer.innerHTML = '';
        if (galleryCard) galleryCard.style.display = 'none';
    }

    // ========== About Section Toggle ==========

    const aboutToggle = document.getElementById('about-toggle');
    const aboutContent = document.getElementById('about-content');
    if (aboutToggle && aboutContent) {
        aboutToggle.addEventListener('click', function() {
            const icon = this.querySelector('.about-toggle-icon');
            if (aboutContent.style.display === 'none') {
                aboutContent.style.display = 'block';
                icon.classList.remove('fa-chevron-down');
                icon.classList.add('fa-chevron-up');
            } else {
                aboutContent.style.display = 'none';
                icon.classList.remove('fa-chevron-up');
                icon.classList.add('fa-chevron-down');
            }
        });
    }

    // ========== Lightbox ==========

    const lightbox = document.getElementById('image-lightbox');
    const lightboxImage = document.getElementById('lightbox-image');
    const lightboxLabel = document.getElementById('lightbox-label');
    const lightboxCommonsLink = document.getElementById('lightbox-commons-link');
    const lightboxClose = document.getElementById('lightbox-close');

    window.openLightbox = function(event, element) {
        event.preventDefault();
        const fullUrl = element.getAttribute('data-full-url');
        const commonsUrl = element.getAttribute('data-commons-url');
        const label = element.getAttribute('data-label');
        const source = element.getAttribute('data-source') || 'wikidata';

        if (lightbox && lightboxImage) {
            lightboxImage.src = fullUrl;
            lightboxLabel.textContent = label;
            lightboxCommonsLink.href = commonsUrl;
            const linkText = lightboxCommonsLink.querySelector('.lightbox-link-text');
            if (linkText) {
                linkText.textContent = source === 'dataset' ? 'View image' : 'View on Wikimedia Commons';
            }
            lightbox.style.display = 'flex';
            document.body.style.overflow = 'hidden';
        }
    };

    if (lightboxClose) {
        lightboxClose.addEventListener('click', function() {
            lightbox.style.display = 'none';
            document.body.style.overflow = '';
            lightboxImage.src = '';
        });
    }

    if (lightbox) {
        lightbox.addEventListener('click', function(e) {
            if (e.target === lightbox) {
                lightbox.style.display = 'none';
                document.body.style.overflow = '';
                lightboxImage.src = '';
            }
        });
    }

    document.addEventListener('keydown', function(e) {
        if (e.key === 'Escape' && lightbox && lightbox.style.display === 'flex') {
            lightbox.style.display = 'none';
            document.body.style.overflow = '';
            lightboxImage.src = '';
        }
    });

    // ========== Inline Entity Thumbnails ==========

    function injectInlineImages(htmlContent, sources) {
        // Build map: lowercase label → { thumbnailUrl, fullUrl, fc, sourceIndex, entityUri, label }
        const labelMap = {};
        sources.forEach((src, idx) => {
            const thumbUrl = EntityCards.getThumbnailUrl(src);
            if (!thumbUrl) return;
            const label = (src.entity_label || '').toLowerCase();
            if (label && !labelMap[label]) {
                labelMap[label] = {
                    thumbnailUrl: thumbUrl,
                    fullUrl: EntityCards.getFullImageUrl(src) || thumbUrl,
                    fc: src.fc || '',
                    sourceIndex: idx,
                    entityUri: src.entity_uri,
                    entityLabel: src.entity_label || ''
                };
            }
        });

        if (Object.keys(labelMap).length === 0) return htmlContent;

        // Parse HTML, walk text nodes, inject thumbnails for first occurrence of each label
        const tempDiv = document.createElement('div');
        tempDiv.innerHTML = htmlContent;

        const injected = new Set();
        const skipTags = new Set(['CODE', 'A', 'PRE', 'IMG', 'SCRIPT', 'STYLE']);

        const walker = document.createTreeWalker(
            tempDiv,
            NodeFilter.SHOW_TEXT,
            {
                acceptNode: function(node) {
                    if (skipTags.has(node.parentElement.tagName)) return NodeFilter.FILTER_REJECT;
                    return NodeFilter.FILTER_ACCEPT;
                }
            }
        );

        const replacements = [];
        let textNode;
        while (textNode = walker.nextNode()) {
            const text = textNode.textContent;
            for (const [label, info] of Object.entries(labelMap)) {
                if (injected.has(label)) continue;
                const idx = text.toLowerCase().indexOf(label);
                if (idx === -1) continue;

                // Mark for injection
                replacements.push({ node: textNode, offset: idx + label.length, info, label });
                injected.add(label);
            }
        }

        // Apply replacements (reverse order to maintain offsets)
        replacements.reverse().forEach(({ node, offset, info }) => {
            const fc = info.fc;
            const fcColor = FC_COLORS[fc] || '#B0BEC5';

            const thumb = document.createElement('img');
            thumb.className = 'inline-entity-thumb';
            thumb.src = info.thumbnailUrl;
            thumb.alt = info.entityLabel;
            thumb.style.border = `2px solid ${fcColor}`;
            thumb.setAttribute('data-entity-uri', info.entityUri);
            thumb.setAttribute('data-full-url', info.fullUrl);
            thumb.setAttribute('data-label', info.entityLabel);
            thumb.onerror = function() { this.style.display = 'none'; };

            // Split text node and insert thumbnail
            const afterText = node.splitText(offset);
            node.parentNode.insertBefore(thumb, afterText);
        });

        return tempDiv.innerHTML;
    }

    // ========== User Message ==========

    function addUserMessage(text) {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message user-message';
        messageDiv.innerHTML = `<p>${text}</p>`;
        chatContainer.appendChild(messageDiv);
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }

    // ========== Assistant Message (redesigned) ==========

    function addAssistantMessage(text, sources = []) {
        const msgId = `msg-${++messageIdCounter}`;

        // Add typing indicator
        const indicatorDiv = document.createElement('div');
        indicatorDiv.className = 'message assistant-message';
        indicatorDiv.innerHTML = `
            <div class="typing-indicator">
                <span></span><span></span><span></span>
            </div>
        `;
        chatContainer.appendChild(indicatorDiv);
        chatContainer.scrollTop = chatContainer.scrollHeight;

        // Process text
        let formattedText = text;
        let ontologySection = '';

        formattedText = formattedText.replace(/\[(\d+)\]/g, (match, p1) => `<sup>[${p1}]</sup>`);

        if (text.includes("Note on CIDOC-CRM concepts used:")) {
            const parts = text.split(/\n\nNote on CIDOC-CRM concepts used:/);
            formattedText = parts[0];
            if (parts.length > 1) {
                ontologySection = `
                    <div class="ontology-info">
                        <h5>CIDOC-CRM Concepts Used:</h5>
                        <p>${parts[1].replace(/\n- /g, '<br/>- ')}</p>
                    </div>
                `;
            }
        }

        setTimeout(() => {
            const messageDiv = document.createElement('div');
            messageDiv.className = 'message assistant-message';
            messageDiv.id = msgId;

            // 1. Answer text with inline thumbnails
            let htmlContent = marked.parse(formattedText);
            if (sources.length > 0) {
                htmlContent = injectInlineImages(htmlContent, sources);
            }

            const answerText = document.createElement('div');
            answerText.className = 'answer-text';
            answerText.innerHTML = htmlContent;
            messageDiv.appendChild(answerText);

            // Ontology section
            if (ontologySection) {
                const ontDiv = document.createElement('div');
                ontDiv.innerHTML = ontologySection;
                messageDiv.appendChild(ontDiv.firstElementChild);
            }

            // 2. Evidence section (mini-graph + entity cards)
            if (sources.length > 0) {
                const evidence = document.createElement('div');
                evidence.className = 'answer-evidence';
                evidence.id = `evidence-${msgId}`;

                // Entity cards strip (shown by default)
                const cardsStrip = document.createElement('div');
                cardsStrip.className = 'entity-cards-strip';
                evidence.appendChild(cardsStrip);

                // Mini-graph (collapsible, hidden by default)
                const miniGraphDetails = document.createElement('details');
                miniGraphDetails.className = 'raw-sources-toggle';
                const miniGraphSummary = document.createElement('summary');
                miniGraphSummary.innerHTML = '<i class="fas fa-project-diagram me-1"></i>Entity graph';
                miniGraphDetails.appendChild(miniGraphSummary);
                const miniGraphWrapper = document.createElement('div');
                miniGraphWrapper.className = 'mini-graph-wrapper';
                miniGraphDetails.appendChild(miniGraphWrapper);
                evidence.appendChild(miniGraphDetails);

                messageDiv.appendChild(evidence);

                // 3. Raw sources toggle (kept for detail access)
                const rawSources = buildRawSourcesToggle(sources);
                if (rawSources) messageDiv.appendChild(rawSources);

                // Replace indicator then render D3 + cards (needs DOM)
                chatContainer.replaceChild(messageDiv, indicatorDiv);
                chatContainer.scrollTop = chatContainer.scrollHeight;

                // Render mini-graph lazily on first open (needs visible width)
                let miniGraphCtrl = null;
                miniGraphDetails.addEventListener('toggle', function onToggle() {
                    if (miniGraphDetails.open && !miniGraphCtrl) {
                        miniGraphCtrl = MiniGraph.renderMiniGraph(miniGraphWrapper, sources);
                        if (miniGraphCtrl) activeMiniGraphs.push(miniGraphCtrl);
                        // Re-wire cross-links with the now-available controller
                        wireCrossLinks(msgId, miniGraphCtrl, cardsCtrl, sources);
                    }
                });

                // Render entity cards
                const cardsCtrl = EntityCards.render(cardsStrip, sources);

                // Update image gallery
                updateImageGallery(sources);

                // 4. Wire cross-linking
                wireCrossLinks(msgId, miniGraphCtrl, cardsCtrl, sources);

                // Wire inline thumbnail clicks → open lightbox
                messageDiv.querySelectorAll('.inline-entity-thumb').forEach(thumb => {
                    thumb.addEventListener('click', function() {
                        const fullUrl = this.getAttribute('data-full-url');
                        const label = this.getAttribute('data-label');
                        if (fullUrl && lightbox && lightboxImage) {
                            lightboxImage.src = fullUrl;
                            lightboxLabel.textContent = label || '';
                            lightboxCommonsLink.href = fullUrl;
                            const linkText = lightboxCommonsLink.querySelector('.lightbox-link-text');
                            if (linkText) linkText.textContent = 'View image';
                            lightbox.style.display = 'flex';
                            document.body.style.overflow = 'hidden';
                        }
                    });
                });
            } else {
                // No sources — simple message
                chatContainer.replaceChild(messageDiv, indicatorDiv);
                chatContainer.scrollTop = chatContainer.scrollHeight;
            }

            // Setup Wikidata + triples buttons
            setupWikidataButtons();
            setupTriplesButtons(sources);

        }, 1200);
    }

    // ========== Cross-Linking Event Wiring ==========

    function wireCrossLinks(msgId, miniGraphCtrl, cardsCtrl, sources) {
        const msgEl = document.getElementById(msgId);
        if (!msgEl) return;

        // Card hover → highlight mini-graph node + gallery image
        msgEl.addEventListener('card-hover', (e) => {
            const { uri, action } = e.detail;
            if (action === 'enter') {
                if (miniGraphCtrl) miniGraphCtrl.highlightNode(uri);
                highlightGalleryImage(uri);
            } else {
                if (miniGraphCtrl) miniGraphCtrl.unhighlightNode(uri);
                unhighlightGalleryImage(uri);
            }
        });

        // Card click → show entity detail panel
        msgEl.addEventListener('card-click', (e) => {
            const { source } = e.detail;
            if (source) showEntityDetail(source);
        });

        // Mini-graph node click → scroll to entity card
        const miniGraphWrapper = msgEl.querySelector('.mini-graph-wrapper');
        if (miniGraphWrapper) {
            miniGraphWrapper.addEventListener('entity-select', (e) => {
                if (cardsCtrl) cardsCtrl.scrollToCard(e.detail.uri);
            });
        }

        // Gallery hover → highlight entity card + mini-graph node
        document.addEventListener('gallery-hover', (e) => {
            const { uri, action } = e.detail;
            if (action === 'enter') {
                if (cardsCtrl) cardsCtrl.highlightCard(uri);
                if (miniGraphCtrl) miniGraphCtrl.highlightNode(uri);
            } else {
                if (cardsCtrl) cardsCtrl.unhighlightCard(uri);
                if (miniGraphCtrl) miniGraphCtrl.unhighlightNode(uri);
            }
        });
    }

    // ========== Raw Sources Toggle (detail panel) ==========

    function buildRawSourcesToggle(sources) {
        if (!sources || sources.length === 0) return null;

        const details = document.createElement('details');
        details.className = 'raw-sources-toggle';

        const summary = document.createElement('summary');
        summary.innerHTML = `<i class="fas fa-database me-1"></i>Raw sources (${sources.length} entities)`;
        details.appendChild(summary);

        const content = document.createElement('div');
        content.style.padding = '8px 0';

        const internalSources = sources.filter(s => s.type === "graph" || s.type === "rdf_data" || s.type === "ontology_documentation");
        const externalSources = sources.filter(s => s.wikidata_id);

        let contentHtml = '';

        if (internalSources.length > 0) {
            contentHtml += '<div class="sources-subsection"><h6>Internal (Graph)</h6><ul class="source-list">';
            internalSources.forEach(source => {
                if (source.type === "graph") {
                    const tripleCount = source.raw_triples ? source.raw_triples.length : 0;
                    const triplesBtnHtml = tripleCount > 0 ?
                        `<button class="btn btn-sm btn-outline-primary show-triples-btn ms-2"
                                data-source-id="${source.id}">
                            <i class="fas fa-code"></i> Show statements (${tripleCount})
                        </button>` : '';
                    const wikidataBtnHtml = source.wikidata_id ?
                        `<button class="btn btn-sm btn-outline-success show-wikidata-btn ms-2"
                                data-entity="${source.entity_uri}"
                                data-wikidata-id="${source.wikidata_id}">
                            <i class="fas fa-info-circle"></i> Wikidata
                        </button>` : '';
                    contentHtml += `<li class="source-item graph-source" data-source-id="${source.id}">
                        <i class="fas fa-project-diagram me-1"></i>
                        <strong>Graph:</strong> ${source.entity_label}
                        <br/>
                        <small class="text-muted ms-3">URI: <code>${source.entity_uri}</code></small>
                        ${triplesBtnHtml}${wikidataBtnHtml}
                    </li>`;
                } else if (source.type === "rdf_data") {
                    contentHtml += `<li class="source-item rdf-source">
                        <a href="/entity/${encodeURIComponent(source.entity_uri)}" target="_blank">
                            ${source.entity_label || source.entity_uri}
                        </a>
                    </li>`;
                } else if (source.type === "ontology_documentation") {
                    contentHtml += `<li class="source-item ontology-source">
                        <a href="/ontology#${source.concept_id}" target="_blank">
                            ${source.concept_id} (${source.concept_name})
                        </a>
                    </li>`;
                }
            });
            contentHtml += '</ul></div>';
        }

        if (externalSources.length > 0) {
            contentHtml += '<div class="sources-subsection"><h6>External (Wikidata)</h6><ul class="source-list">';
            externalSources.forEach(source => {
                contentHtml += `<li class="source-item wikidata-source">
                    <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/f/ff/Wikidata-logo.svg/20px-Wikidata-logo.svg.png" class="wikidata-logo" alt="Wikidata">
                    <a href="${source.wikidata_url}" target="_blank">
                        ${source.entity_label} (Wikidata)
                    </a>
                    <button class="btn btn-sm btn-outline-success show-wikidata-btn"
                            data-entity="${source.entity_uri}"
                            data-wikidata-id="${source.wikidata_id}">
                        <i class="fas fa-info-circle"></i> More info
                    </button>
                </li>`;
            });
            contentHtml += '</ul></div>';
        }

        content.innerHTML = contentHtml;
        details.appendChild(content);
        return details;
    }

    // ========== Wikidata Buttons ==========

    function setupWikidataButtons() {
        document.querySelectorAll('.show-wikidata-btn').forEach(button => {
            if (!button.hasListener) {
                button.addEventListener('click', async function(e) {
                    e.preventDefault();
                    const entityUri = this.getAttribute('data-entity');
                    const wikidataId = this.getAttribute('data-wikidata-id');

                    const parentLi = this.closest('li');
                    const loadingDiv = document.createElement('div');
                    loadingDiv.className = 'text-muted small mt-1';
                    loadingDiv.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Loading Wikidata information...';
                    parentLi.appendChild(loadingDiv);

                    try {
                        let wikidataUrl = `/api/entity/${encodeURIComponent(entityUri)}/wikidata`;
                        if (currentDatasetId) wikidataUrl += `?dataset_id=${encodeURIComponent(currentDatasetId)}`;
                        const response = await fetch(wikidataUrl);
                        if (!response.ok) throw new Error('Failed to fetch Wikidata information');

                        const data = await response.json();
                        parentLi.removeChild(loadingDiv);

                        const wikidataPanel = document.createElement('div');
                        wikidataPanel.className = 'wikidata-panel mt-2';

                        let panelContent = `
                            <div class="wikidata-title">
                                <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/f/ff/Wikidata-logo.svg/20px-Wikidata-logo.svg.png" class="wikidata-logo" alt="Wikidata">
                                ${data.label || 'Wikidata Entity'}
                            </div>
                        `;

                        if (data.description) panelContent += `<p>${data.description}</p>`;

                        if (data.properties) {
                            panelContent += '<ul class="list-unstyled mb-0">';
                            for (const [propName, propValue] of Object.entries(data.properties)) {
                                const formattedPropName = propName.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
                                if (propName === 'coordinates' && propValue.latitude && propValue.longitude) {
                                    panelContent += `<li><strong>${formattedPropName}:</strong> ${propValue.latitude.toFixed(4)}, ${propValue.longitude.toFixed(4)}</li>`;
                                } else if (propName === 'image') {
                                    const imageFilename = Array.isArray(propValue) ? propValue[0] : propValue;
                                    if (imageFilename && typeof imageFilename === 'string') {
                                        const encodedFilename = encodeURIComponent(imageFilename);
                                        const thumbnailUrl = `https://commons.wikimedia.org/wiki/Special:FilePath/${encodedFilename}?width=250`;
                                        const commonsUrl = `https://commons.wikimedia.org/wiki/File:${encodedFilename}`;
                                        panelContent += `
                                            <li class="wikidata-image-property">
                                                <strong>${formattedPropName}:</strong>
                                                <div class="wikidata-panel-image-container">
                                                    <a href="${commonsUrl}" target="_blank" title="View on Wikimedia Commons">
                                                        <img src="${thumbnailUrl}"
                                                             alt="${data.label || 'Image'}"
                                                             class="wikidata-panel-thumbnail"
                                                             loading="lazy"
                                                             onerror="this.parentElement.parentElement.style.display='none'">
                                                    </a>
                                                </div>
                                            </li>`;
                                    }
                                } else if (Array.isArray(propValue)) {
                                    panelContent += `<li><strong>${formattedPropName}:</strong> ${propValue.join(', ')}</li>`;
                                } else {
                                    panelContent += `<li><strong>${formattedPropName}:</strong> ${propValue}</li>`;
                                }
                            }
                            panelContent += '</ul>';
                        }

                        if (data.wikipedia) {
                            panelContent += `
                                <div class="mt-2">
                                    <a href="${data.wikipedia.url}" target="_blank" class="btn btn-sm btn-outline-secondary">
                                        <i class="fab fa-wikipedia-w"></i> Read on Wikipedia
                                    </a>
                                    <a href="${data.url}" target="_blank" class="btn btn-sm btn-outline-success ml-2">
                                        <i class="fas fa-database"></i> View on Wikidata
                                    </a>
                                </div>
                            `;
                        }

                        wikidataPanel.innerHTML = panelContent;
                        parentLi.appendChild(wikidataPanel);
                        this.style.display = 'none';
                    } catch (error) {
                        console.error('Error fetching Wikidata:', error);
                        parentLi.removeChild(loadingDiv);
                        const errorDiv = document.createElement('div');
                        errorDiv.className = 'text-danger small mt-1';
                        errorDiv.textContent = 'Failed to load Wikidata information.';
                        parentLi.appendChild(errorDiv);
                    }
                });
                button.hasListener = true;
            }
        });
    }

    // ========== Triples Buttons ==========

    function setupTriplesButtons(sources) {
        document.querySelectorAll('.show-triples-btn').forEach(button => {
            if (!button.hasListener) {
                button.addEventListener('click', function(e) {
                    e.preventDefault();
                    const sourceId = parseInt(this.getAttribute('data-source-id'));
                    const source = sources.find(s => s.id === sourceId);
                    if (!source || !source.raw_triples || source.raw_triples.length === 0) return;

                    const parentLi = this.closest('li');
                    const existingPanel = parentLi.querySelector('.triples-panel');
                    if (existingPanel) {
                        existingPanel.remove();
                        this.innerHTML = `<i class="fas fa-code"></i> Show statements (${source.raw_triples.length})`;
                    } else {
                        const triplesPanel = document.createElement('div');
                        triplesPanel.className = 'triples-panel mt-2';

                        let panelContent = `
                            <div class="triples-header">
                                <strong>RDF Statements</strong>
                                <span class="text-muted">(${source.raw_triples.length} triples)</span>
                            </div>
                            <div class="triples-list">
                        `;

                        source.raw_triples.forEach(triple => {
                            panelContent += `
                                <div class="triple-item">
                                    <div class="triple-part triple-subject">
                                        <span class="triple-label">Subject:</span>
                                        <span class="triple-value" title="${triple.subject}">${triple.subject_label || triple.subject}</span>
                                    </div>
                                    <div class="triple-part triple-predicate">
                                        <span class="triple-label">Predicate:</span>
                                        <span class="triple-value" title="${triple.predicate}">${triple.predicate_label || triple.predicate}</span>
                                    </div>
                                    <div class="triple-part triple-object">
                                        <span class="triple-label">Object:</span>
                                        <span class="triple-value" title="${triple.object}">${triple.object_label || triple.object}</span>
                                    </div>
                                </div>
                            `;
                        });

                        panelContent += '</div>';
                        triplesPanel.innerHTML = panelContent;
                        parentLi.appendChild(triplesPanel);
                        this.innerHTML = `<i class="fas fa-code"></i> Hide statements`;
                    }
                });
                button.hasListener = true;
            }
        });
    }

    // ========== Send Question ==========

    async function sendQuestion(question) {
        if (!question.trim()) return;

        if (!currentDatasetId && datasetSelect.options.length > 1) {
            addAssistantMessage('Please select a dataset from the dropdown before asking questions.');
            return;
        }

        addUserMessage(question);
        clearImageGallery();
        hideEntityDetail();

        // Auto-collapse exploration panel after first question to free sidebar space
        if (!explorationCollapsed) {
            setExplorationCollapsed(true);
        }

        questionInput.value = '';
        sendButton.disabled = true;

        chatHistory.push({ role: 'user', content: question });

        try {
            const requestBody = { question };
            if (currentDatasetId) requestBody.dataset_id = currentDatasetId;
            if (chatHistory.length > 1) requestBody.chat_history = chatHistory.slice(-6);

            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(requestBody),
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || `Server responded with status: ${response.status}`);
            }

            const data = await response.json();
            chatHistory.push({ role: 'assistant', content: data.answer });
            addAssistantMessage(data.answer, data.sources);
        } catch (error) {
            console.error('Error:', error);
            addAssistantMessage(`Sorry, I encountered an error: ${error.message}`);
        } finally {
            sendButton.disabled = false;
            questionInput.focus();
        }
    }

    // ========== Event Listeners ==========

    sendButton.addEventListener('click', () => sendQuestion(questionInput.value));

    questionInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') sendQuestion(questionInput.value);
    });

    exampleQuestions.forEach(question => {
        question.addEventListener('click', (e) => {
            e.preventDefault();
            questionInput.value = question.textContent;
            sendQuestion(question.textContent);
        });
    });

    questionInput.focus();
});
