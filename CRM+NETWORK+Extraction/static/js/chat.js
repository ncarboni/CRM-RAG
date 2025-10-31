document.addEventListener('DOMContentLoaded', function() {
    const chatContainer = document.getElementById('chat-container');
    const questionInput = document.getElementById('question-input');
    const sendButton = document.getElementById('send-button');
    const exampleQuestions = document.querySelectorAll('.example-question');

    // Fetch system info and update greeting
    async function loadSystemInfo() {
        try {
            const response = await fetch('/api/info');
            if (response.ok) {
                const data = await response.json();
                // Update the initial greeting with dataset description
                const initialMessage = chatContainer.querySelector('.assistant-message p');
                if (initialMessage && data.dataset_description) {
                    initialMessage.textContent = `Hello! I'm a chatbot assistant. I can answer questions about ${data.dataset_description}. How can I help you today?`;
                }
            }
        } catch (error) {
            console.error('Error loading system info:', error);
        }
    }

    // Load system info on page load
    loadSystemInfo();
    
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
                sourcesDiv.innerHTML = '<p><strong>Sources:</strong></p><ul class="source-list">';
                
                // Track if we have Wikidata sources
                let hasWikidataSources = false;
                
                sources.forEach(source => {
                    if (source.type === "rdf_data") {
                        sourcesDiv.innerHTML += `<li class="source-item rdf-source">
                            <a href="/entity/${encodeURIComponent(source.entity_uri)}" target="_blank">
                                ${source.entity_label || source.entity_uri}
                            </a>
                        </li>`;
                    } else if (source.type === "ontology_documentation") {
                        sourcesDiv.innerHTML += `<li class="source-item ontology-source">
                            <a href="/ontology#${source.concept_id}" target="_blank">
                                ${source.concept_id} (${source.concept_name})
                            </a>
                        </li>`;
                    } else if (source.type === "wikidata") {
                        hasWikidataSources = true;
                        sourcesDiv.innerHTML += `<li class="source-item wikidata-source">
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
                    }
                });
                
                sourcesDiv.innerHTML += '</ul>';
                
                // If we have Wikidata sources, add an explanation
                if (hasWikidataSources) {
                    sourcesDiv.innerHTML += '<div class="text-muted small mt-2">Wikidata sources provide additional context from external data sources.</div>';
                }
                
                messageDiv.appendChild(sourcesDiv);
            }
            
            // Replace typing indicator with actual message
            chatContainer.replaceChild(messageDiv, indicatorDiv);
            chatContainer.scrollTop = chatContainer.scrollHeight;
            
            // Set up Wikidata buttons
            setupWikidataButtons();
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
                        // Fetch Wikidata information
                        const response = await fetch(`/api/entity/${encodeURIComponent(entityUri)}/wikidata`);
                        
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
                                    // Don't include image URLs directly
                                    continue;
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
    
    // Function to send question to server
    async function sendQuestion(question) {
        // Don't send empty questions
        if (!question.trim()) return;
        
        // Add user message to chat
        addUserMessage(question);
        
        // Clear input and disable send button
        questionInput.value = '';
        sendButton.disabled = true;
        
        try {
            // Send request to server
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ question }),
            });
            
            if (!response.ok) {
                throw new Error(`Server responded with status: ${response.status}`);
            }
            
            const data = await response.json();
            
            // Add assistant response to chat
            addAssistantMessage(data.answer, data.sources);
        } catch (error) {
            console.error('Error:', error);
            addAssistantMessage('Sorry, I encountered an error while processing your question. Please try again later.');
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