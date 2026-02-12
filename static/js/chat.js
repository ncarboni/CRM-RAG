document.addEventListener('DOMContentLoaded', function() {
    const chatContainer = document.getElementById('chat-container');
    const questionInput = document.getElementById('question-input');
    const sendButton = document.getElementById('send-button');
    let exampleQuestions = document.querySelectorAll('.example-question');

    // Dataset management state
    let currentDatasetId = null;
    // Conversation history for follow-up context
    let chatHistory = [];
    const datasetSelect = document.getElementById('dataset-select');
    const datasetStatus = document.getElementById('dataset-status');

    // Load available datasets on page load
    async function loadDatasets() {
        try {
            const response = await fetch('/api/datasets');
            if (response.ok) {
                const data = await response.json();

                // Clear existing options except the placeholder
                datasetSelect.innerHTML = '<option value="" disabled selected>Select Dataset...</option>';

                // Add datasets to dropdown
                data.datasets.forEach(ds => {
                    const option = document.createElement('option');
                    option.value = ds.id;
                    let displayText = ds.display_name;
                    if (!ds.has_cache) {
                        displayText += ' (not built)';
                    } else if (!ds.initialized) {
                        displayText += ' (not loaded)';
                    }
                    option.textContent = displayText;
                    datasetSelect.appendChild(option);
                });

                // Auto-select default dataset if set
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

    // Switch to a different dataset
    async function switchDataset(datasetId) {
        if (!datasetId) return;

        // Update status to loading
        datasetStatus.textContent = 'Loading...';
        datasetStatus.className = 'ms-2 badge bg-warning';
        sendButton.disabled = true;

        try {
            const response = await fetch(`/api/datasets/${datasetId}/select`, {
                method: 'POST'
            });

            if (response.ok) {
                const data = await response.json();
                currentDatasetId = datasetId;

                // Update interface elements
                updateInterface(data.interface);

                // Clear chat and show welcome message
                clearChat();
                addWelcomeMessage(data.interface.welcome_message);

                // Update status
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

    // Update interface elements based on dataset config
    function updateInterface(config) {
        // Update page title
        if (config.page_title) {
            document.title = config.page_title;
        }

        // Update header title
        const headerTitle = document.getElementById('header-title');
        if (headerTitle && config.header_title) {
            headerTitle.innerHTML = `<i class="fas fa-comments me-2"></i>${config.header_title}`;
        }

        // Update input placeholder
        if (config.input_placeholder) {
            questionInput.placeholder = config.input_placeholder;
        }

        // Update example questions
        updateExampleQuestions(config.example_questions);
    }

    // Update the example questions section
    function updateExampleQuestions(questions) {
        if (!questions || questions.length === 0) return;

        const container = document.getElementById('example-questions-row');
        if (!container) return;

        // Split questions into two columns
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

        // Re-bind example question click handlers
        exampleQuestions = document.querySelectorAll('.example-question');
        exampleQuestions.forEach(question => {
            question.addEventListener('click', (e) => {
                e.preventDefault();
                questionInput.value = question.textContent;
                sendQuestion(question.textContent);
            });
        });
    }

    // Clear the chat container
    function clearChat() {
        chatContainer.innerHTML = '';
    }

    // Add welcome message
    function addWelcomeMessage(message) {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message assistant-message';
        messageDiv.innerHTML = `<p>${message}</p>`;
        chatContainer.appendChild(messageDiv);
    }

    // Handle dataset selection change
    datasetSelect.addEventListener('change', (e) => {
        switchDataset(e.target.value);
    });

    // Load datasets on page load
    loadDatasets();

    // Fetch system info and update greeting (fallback for single-dataset mode)
    async function loadSystemInfo() {
        try {
            const response = await fetch('/api/info');
            if (response.ok) {
                const data = await response.json();
                // Only update if no dataset is selected (single-dataset mode)
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

    // Load system info on page load (for single-dataset mode fallback)
    loadSystemInfo();

    // Function to update the image gallery panel
    function updateImageGallery(sources) {
        const galleryContainer = document.getElementById('image-gallery');
        const galleryColumn = document.getElementById('image-gallery-column');
        if (!galleryContainer || !galleryColumn) return;

        // Collect all images from sources (deduplicate by URL)
        const imageMap = new Map();
        sources.forEach(source => {
            // Handle Wikidata images (source.image — singular, has thumbnail_url)
            if (source.image && source.image.thumbnail_url) {
                // Validate the URL - skip if it looks malformed (e.g., array stringified)
                const thumbnailUrl = source.image.thumbnail_url;
                if (typeof thumbnailUrl === 'string' &&
                    thumbnailUrl.startsWith('http') &&
                    !thumbnailUrl.includes('[') &&
                    !thumbnailUrl.includes(']')) {

                    const key = thumbnailUrl;
                    if (!imageMap.has(key)) {
                        imageMap.set(key, {
                            thumbnailUrl: thumbnailUrl,
                            fullUrl: source.image.full_url || thumbnailUrl,
                            linkUrl: source.image.url || thumbnailUrl,
                            label: source.entity_label || 'Image',
                            source: source.image.source || 'wikidata'
                        });
                    }
                }
            }
            // Handle dataset images (source.images — plural, array of {url, source})
            if (source.images && Array.isArray(source.images)) {
                source.images.forEach(img => {
                    const imgUrl = img.url;
                    if (typeof imgUrl === 'string' &&
                        imgUrl.startsWith('http') &&
                        !imageMap.has(imgUrl)) {
                        imageMap.set(imgUrl, {
                            thumbnailUrl: imgUrl,
                            fullUrl: imgUrl,
                            linkUrl: imgUrl,
                            label: source.entity_label || 'Image',
                            source: img.source || 'dataset'
                        });
                    }
                });
            }
        });
        const images = Array.from(imageMap.values());

        // If no images, hide the gallery column
        if (images.length === 0) {
            galleryColumn.style.display = 'none';
            galleryContainer.innerHTML = '';
            return;
        }

        // Show the gallery column
        galleryColumn.style.display = 'block';

        // Build gallery HTML
        let galleryHtml = '<div class="image-gallery-grid" id="image-gallery-grid">';
        images.forEach((img, index) => {
            galleryHtml += `
                <div class="gallery-item" data-image-index="${index}">
                    <a href="#" class="gallery-image-link"
                       data-full-url="${img.fullUrl}"
                       data-commons-url="${img.linkUrl}"
                       data-label="${img.label}"
                       data-source="${img.source}"
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
                    </a>
                </div>
            `;
        });
        galleryHtml += '</div>';

        // Add image count (will be updated if images fail to load)
        galleryHtml += `<div class="gallery-count text-muted small mt-2" id="gallery-count">${images.length} image${images.length > 1 ? 's' : ''} found</div>`;

        galleryContainer.innerHTML = galleryHtml;
    }

    // Global function to handle image load errors
    window.handleImageError = function(imgElement) {
        const galleryItem = imgElement.closest('.gallery-item');
        if (galleryItem) {
            galleryItem.style.display = 'none';

            // Update the count and check if all images failed
            const galleryGrid = document.getElementById('image-gallery-grid');
            const galleryCount = document.getElementById('gallery-count');
            const galleryColumn = document.getElementById('image-gallery-column');

            if (galleryGrid) {
                const visibleItems = galleryGrid.querySelectorAll('.gallery-item:not([style*="display: none"])');
                const count = visibleItems.length;

                if (count === 0) {
                    // All images failed, hide the gallery
                    if (galleryColumn) {
                        galleryColumn.style.display = 'none';
                    }
                } else if (galleryCount) {
                    // Update the count
                    galleryCount.textContent = `${count} image${count > 1 ? 's' : ''} found`;
                }
            }
        }
    };

    // Function to clear/hide the image gallery
    function clearImageGallery() {
        const galleryContainer = document.getElementById('image-gallery');
        const galleryColumn = document.getElementById('image-gallery-column');
        if (galleryContainer) {
            galleryContainer.innerHTML = '';
        }
        if (galleryColumn) {
            galleryColumn.style.display = 'none';
        }
    }

    // Setup About section toggle
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

    // Setup lightbox functionality
    const lightbox = document.getElementById('image-lightbox');
    const lightboxImage = document.getElementById('lightbox-image');
    const lightboxLabel = document.getElementById('lightbox-label');
    const lightboxCommonsLink = document.getElementById('lightbox-commons-link');
    const lightboxClose = document.getElementById('lightbox-close');

    // Global function to open lightbox (called from onclick)
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

    // Close lightbox on close button click
    if (lightboxClose) {
        lightboxClose.addEventListener('click', function() {
            lightbox.style.display = 'none';
            document.body.style.overflow = '';
            lightboxImage.src = '';
        });
    }

    // Close lightbox on background click
    if (lightbox) {
        lightbox.addEventListener('click', function(e) {
            if (e.target === lightbox) {
                lightbox.style.display = 'none';
                document.body.style.overflow = '';
                lightboxImage.src = '';
            }
        });
    }

    // Close lightbox on Escape key
    document.addEventListener('keydown', function(e) {
        if (e.key === 'Escape' && lightbox && lightbox.style.display === 'flex') {
            lightbox.style.display = 'none';
            document.body.style.overflow = '';
            lightboxImage.src = '';
        }
    });

    // Function to add user message
    function addUserMessage(text) {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message user-message';
        messageDiv.innerHTML = `<p>${text}</p>`;
        chatContainer.appendChild(messageDiv);
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }
    
    // Function to add assistant message
    function addAssistantMessage(text, sources = []) {
        // Add typing indicator first
        const indicatorDiv = document.createElement('div');
        indicatorDiv.className = 'message assistant-message';
        indicatorDiv.innerHTML = `
            <div class="typing-indicator">
                <span></span>
                <span></span>
                <span></span>
            </div>
        `;
        chatContainer.appendChild(indicatorDiv);
        chatContainer.scrollTop = chatContainer.scrollHeight;
        
        // Process text to handle markdown-style citations and ontology section
        let formattedText = text;
        let ontologySection = '';
        
        // Handle citations
        formattedText = formattedText.replace(/\[(\d+)\]/g, (match, p1) => {
            return `<sup>[${p1}]</sup>`;
        });
        
        // Extract and format ontology section if present
        if (text.includes("Note on CIDOC-CRM concepts used:")) {
            const parts = text.split(/\n\nNote on CIDOC-CRM concepts used:/);
            formattedText = parts[0];
            
            if (parts.length > 1) {
                const ontologyInfo = parts[1];
                ontologySection = `
                    <div class="ontology-info">
                        <h5>CIDOC-CRM Concepts Used:</h5>
                        <p>${ontologyInfo.replace(/\n- /g, '<br/>- ')}</p>
                    </div>
                `;
            }
        }
        
        // Replace the indicator with the actual message after a short delay
        setTimeout(() => {
            // Create the actual message
            const messageDiv = document.createElement('div');
            messageDiv.className = 'message assistant-message';
            
            // Convert markdown to HTML
            const htmlContent = marked.parse(formattedText);
            
            // Add the message text
            messageDiv.innerHTML = `<div>${htmlContent}</div>`;
            
            // Add ontology section if present
            if (ontologySection) {
                messageDiv.innerHTML += ontologySection;
            }
            
            // Add sources if any
            if (sources && sources.length > 0) {
                const sourcesDiv = document.createElement('div');
                sourcesDiv.className = 'sources-section';

                // Separate sources into internal and external
                const internalSources = sources.filter(s => s.type === "graph" || s.type === "rdf_data" || s.type === "ontology_documentation");
                const externalSources = sources.filter(s => s.wikidata_id);

                // Create collapsible toggle
                const toggleDiv = document.createElement('div');
                toggleDiv.className = 'sources-toggle';
                toggleDiv.innerHTML = `
                    <i class="fas fa-chevron-right sources-toggle-icon"></i>
                    <span>Sources</span>
                `;

                // Create collapsible content
                const contentDiv = document.createElement('div');
                contentDiv.className = 'sources-content';

                let contentHtml = '';

                // Add internal sources section
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

                // Add external sources section
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

                    contentHtml += '</ul>';
                    contentHtml += '<div class="text-muted small mt-2">Wikidata sources provide additional context from external data sources.</div>';
                    contentHtml += '</div>';
                }

                // Update the image gallery with images from all sources
                updateImageGallery(sources);

                contentDiv.innerHTML = contentHtml;

                // Add toggle functionality
                toggleDiv.addEventListener('click', function() {
                    const icon = this.querySelector('.sources-toggle-icon');
                    const content = this.nextElementSibling;

                    icon.classList.toggle('expanded');
                    content.classList.toggle('expanded');
                });

                sourcesDiv.appendChild(toggleDiv);
                sourcesDiv.appendChild(contentDiv);
                messageDiv.appendChild(sourcesDiv);
            }
            
            // Replace typing indicator with actual message
            chatContainer.replaceChild(messageDiv, indicatorDiv);
            chatContainer.scrollTop = chatContainer.scrollHeight;

            // Set up Wikidata buttons
            setupWikidataButtons();

            // Set up triples buttons
            setupTriplesButtons(sources);
        }, 1500); // Simulated typing delay
    }
    
    // Function to handle Wikidata info buttons
    function setupWikidataButtons() {
        document.querySelectorAll('.show-wikidata-btn').forEach(button => {
            if (!button.hasListener) {
                button.addEventListener('click', async function(e) {
                    e.preventDefault();
                    const entityUri = this.getAttribute('data-entity');
                    const wikidataId = this.getAttribute('data-wikidata-id');
                    
                    // Show loading indicator
                    const parentLi = this.closest('li');
                    const loadingDiv = document.createElement('div');
                    loadingDiv.className = 'text-muted small mt-1';
                    loadingDiv.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Loading Wikidata information...';
                    parentLi.appendChild(loadingDiv);
                    
                    try {
                        // Fetch Wikidata information (include dataset_id if available)
                        let wikidataUrl = `/api/entity/${encodeURIComponent(entityUri)}/wikidata`;
                        if (currentDatasetId) {
                            wikidataUrl += `?dataset_id=${encodeURIComponent(currentDatasetId)}`;
                        }
                        const response = await fetch(wikidataUrl);
                        
                        if (!response.ok) {
                            throw new Error('Failed to fetch Wikidata information');
                        }
                        
                        const data = await response.json();
                        
                        // Remove loading indicator
                        parentLi.removeChild(loadingDiv);
                        
                        // Create Wikidata panel
                        const wikidataPanel = document.createElement('div');
                        wikidataPanel.className = 'wikidata-panel mt-2';
                        
                        let panelContent = `
                            <div class="wikidata-title">
                                <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/f/ff/Wikidata-logo.svg/20px-Wikidata-logo.svg.png" class="wikidata-logo" alt="Wikidata">
                                ${data.label || 'Wikidata Entity'}
                            </div>
                        `;
                        
                        if (data.description) {
                            panelContent += `<p>${data.description}</p>`;
                        }
                        
                        if (data.properties) {
                            panelContent += '<ul class="list-unstyled mb-0">';
                            
                            for (const [propName, propValue] of Object.entries(data.properties)) {
                                const formattedPropName = propName.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());

                                if (propName === 'coordinates' && propValue.latitude && propValue.longitude) {
                                    panelContent += `<li><strong>${formattedPropName}:</strong> ${propValue.latitude.toFixed(4)}, ${propValue.longitude.toFixed(4)}</li>`;
                                } else if (propName === 'image') {
                                    // Display image with link to Wikimedia Commons
                                    // Handle both single image and array of images
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
                        
                        // Hide the button
                        this.style.display = 'none';
                        
                    } catch (error) {
                        console.error('Error fetching Wikidata:', error);
                        // Remove loading indicator
                        parentLi.removeChild(loadingDiv);
                        
                        // Show error message
                        const errorDiv = document.createElement('div');
                        errorDiv.className = 'text-danger small mt-1';
                        errorDiv.textContent = 'Failed to load Wikidata information.';
                        parentLi.appendChild(errorDiv);
                    }
                });
                
                // Mark that we added a listener to avoid duplicates
                button.hasListener = true;
            }
        });
    }

    // Function to handle triples/statements buttons
    function setupTriplesButtons(sources) {
        document.querySelectorAll('.show-triples-btn').forEach(button => {
            if (!button.hasListener) {
                button.addEventListener('click', function(e) {
                    e.preventDefault();
                    const sourceId = parseInt(this.getAttribute('data-source-id'));

                    // Find the source in the sources array
                    const source = sources.find(s => s.id === sourceId);
                    if (!source || !source.raw_triples || source.raw_triples.length === 0) {
                        return;
                    }

                    const parentLi = this.closest('li');

                    // Check if triples are already shown
                    const existingPanel = parentLi.querySelector('.triples-panel');
                    if (existingPanel) {
                        // Hide the panel
                        existingPanel.remove();
                        this.innerHTML = `<i class="fas fa-code"></i> Show statements (${source.raw_triples.length})`;
                    } else {
                        // Show the panel
                        const triplesPanel = document.createElement('div');
                        triplesPanel.className = 'triples-panel mt-2';

                        let panelContent = `
                            <div class="triples-header">
                                <strong>RDF Statements</strong>
                                <span class="text-muted">(${source.raw_triples.length} triples)</span>
                            </div>
                            <div class="triples-list">
                        `;

                        source.raw_triples.forEach((triple, idx) => {
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

                        // Update button text
                        this.innerHTML = `<i class="fas fa-code"></i> Hide statements`;
                    }
                });

                // Mark that we added a listener to avoid duplicates
                button.hasListener = true;
            }
        });
    }

    // Function to send question to server
    async function sendQuestion(question) {
        // Don't send empty questions
        if (!question.trim()) return;

        // Check if a dataset is selected (in multi-dataset mode)
        if (!currentDatasetId && datasetSelect.options.length > 1) {
            addAssistantMessage('Please select a dataset from the dropdown before asking questions.');
            return;
        }

        // Add user message to chat
        addUserMessage(question);

        // Clear image gallery while loading new response
        clearImageGallery();

        // Clear input and disable send button
        questionInput.value = '';
        sendButton.disabled = true;

        // Track user message in history
        chatHistory.push({ role: 'user', content: question });

        try {
            // Build request body with dataset_id and conversation history
            const requestBody = { question };
            if (currentDatasetId) {
                requestBody.dataset_id = currentDatasetId;
            }
            // Send last 6 messages (3 exchanges) for context
            if (chatHistory.length > 1) {
                requestBody.chat_history = chatHistory.slice(-6);
            }

            // Send request to server
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(requestBody),
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || `Server responded with status: ${response.status}`);
            }

            const data = await response.json();

            // Track assistant response in history
            chatHistory.push({ role: 'assistant', content: data.answer });

            // Add assistant response to chat
            addAssistantMessage(data.answer, data.sources);
        } catch (error) {
            console.error('Error:', error);
            addAssistantMessage(`Sorry, I encountered an error: ${error.message}`);
        } finally {
            // Re-enable send button
            sendButton.disabled = false;
            // Focus input for next question
            questionInput.focus();
        }
    }
    
    // Handle send button click
    sendButton.addEventListener('click', () => {
        sendQuestion(questionInput.value);
    });
    
    // Handle Enter key press
    questionInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            sendQuestion(questionInput.value);
        }
    });
    
    // Handle example questions
    exampleQuestions.forEach(question => {
        question.addEventListener('click', (e) => {
            e.preventDefault();
            questionInput.value = question.textContent;
            sendQuestion(question.textContent);
        });
    });
    
    // Focus input field on load
    questionInput.focus();
});