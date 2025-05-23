let selectedDocuments = new Set();

// Update initialization status display
function updateInitializationUI(data) {
    const {
        is_initialized,
        status,
        error,
        progress
    } = data;

    // Update status message
    document.getElementById('initStatusMessage').textContent = progress.detailed_status || status;
    
    // Update progress bar
    const progressBar = document.getElementById('progressBar');
    const progressPercent = document.getElementById('progressPercent');
    const currentStep = document.getElementById('currentStep');
    
    progressBar.style.width = `${progress.download_progress}%`;
    progressPercent.textContent = `${progress.download_progress}%`;
    currentStep.textContent = progress.current_step;

    // Update step indicators
    const steps = document.querySelectorAll('#initSteps li');
    const currentStepNumber = progress.current_step_number;
    
    steps.forEach((step, index) => {
        const indicator = step.querySelector('span');
        if (index < currentStepNumber) {
            // Completed step
            indicator.className = 'w-4 h-4 mr-2 rounded-full bg-green-500';
            step.classList.remove('text-gray-500');
            step.classList.add('text-green-700');
        } else if (index === currentStepNumber) {
            // Current step
            indicator.className = 'w-4 h-4 mr-2 rounded-full bg-blue-500';
            step.classList.remove('text-gray-500');
            step.classList.add('text-blue-700');
        }
    });

    // Handle errors
    const errorDiv = document.getElementById('initError');
    if (error) {
        errorDiv.textContent = `Initialization Error: ${error}`;
        errorDiv.classList.remove('hidden');
    } else {
        errorDiv.classList.add('hidden');
    }

    // Show/hide main content
    if (is_initialized) {
        document.getElementById('initStatus').classList.add('hidden');
        document.getElementById('mainContent').classList.remove('hidden');
    } else {
        document.getElementById('initStatus').classList.remove('hidden');
        document.getElementById('mainContent').classList.add('hidden');
    }
}

// Check initialization status on page load
async function checkInitStatus() {
    try {
        const response = await fetch('/status');
        const data = await response.json();
        updateInitializationUI(data);
        
        if (!data.is_initialized) {
            // Check again in 500ms for more responsive updates
            setTimeout(checkInitStatus, 500);
        }
    } catch (error) {
        console.error('Error checking initialization status:', error);
        // On error, retry after 1 second
        setTimeout(checkInitStatus, 1000);
    }
}

// Start initialization and status checking
async function initializeSystem() {
    try {
        // Show initialization status immediately
        document.getElementById('initStatus').classList.remove('hidden');
        document.getElementById('mainContent').classList.add('hidden');
        
        // Reset progress indicators
        document.getElementById('progressBar').style.width = '0%';
        document.getElementById('progressPercent').textContent = '0%';
        document.getElementById('currentStep').textContent = 'Starting initialization...';
        
        // Reset step indicators
        const steps = document.querySelectorAll('#initSteps li');
        steps.forEach(step => {
            const indicator = step.querySelector('span');
            indicator.className = 'w-4 h-4 mr-2 rounded-full border-2 border-gray-300';
            step.classList.remove('text-green-700', 'text-blue-700');
            step.classList.add('text-gray-500');
        });
        
        // Start initialization
        await fetch('/initialize', { method: 'POST' });
        checkInitStatus();
    } catch (error) {
        console.error('Error starting initialization:', error);
        document.getElementById('initError').textContent = `Initialization Error: ${error.message}`;
        document.getElementById('initError').classList.remove('hidden');
    }
}

// Handle document checkbox changes
function handleDocumentCheckbox(checkbox) {
    if (checkbox.checked) {
        selectedDocuments.add(checkbox.value);
    } else {
        selectedDocuments.delete(checkbox.value);
    }
    
    // Enable/disable start chat button based on selection
    const startChatButton = document.getElementById('startChatButton');
    if (startChatButton) {
        startChatButton.disabled = selectedDocuments.size === 0;
    }
}

// Start chat with selected documents
function startChat() {
    if (selectedDocuments.size === 0) return;
    
    const chatSection = document.getElementById('chatSection');
    if (chatSection) {
        chatSection.style.display = 'block';
    }
    
    const chatMessages = document.getElementById('chatMessages');
    if (chatMessages) {
        chatMessages.innerHTML = '';
    }
    
    // Display selected documents
    const selectedDocsDiv = document.getElementById('selectedDocuments');
    if (selectedDocsDiv) {
        const docElements = Array.from(selectedDocuments).map(docId => {
            const checkbox = document.querySelector(`input[value="${docId}"]`);
            const filename = checkbox?.nextElementSibling?.textContent || docId;
            return `<span class="inline-block bg-blue-100 text-blue-800 px-2 py-1 rounded mr-2 mb-2">${filename}</span>`;
        });
        selectedDocsDiv.innerHTML = '<div class="font-medium mb-2">Selected Documents:</div>' + docElements.join('');
    }
    
    addMessage('system', 'Documents selected. You can now start chatting.');
}

// Close chat and reset selection
function closeChat() {
    const chatSection = document.getElementById('chatSection');
    const chatMessages = document.getElementById('chatMessages');
    const selectedDocsDiv = document.getElementById('selectedDocuments');
    const startChatButton = document.getElementById('startChatButton');
    
    if (chatSection) {
        chatSection.style.display = 'none';
    }
    if (chatMessages) {
        chatMessages.innerHTML = '';
    }
    if (selectedDocsDiv) {
        selectedDocsDiv.innerHTML = '';
    }
    
    // Uncheck all checkboxes
    selectedDocuments.clear();
    document.querySelectorAll('input[type="checkbox"]').forEach(cb => {
        cb.checked = false;
    });
    
    if (startChatButton) {
        startChatButton.disabled = true;
    }
}

// Delete document
async function deleteDocument(documentId, filename) {
    if (!confirm(`Are you sure you want to delete "${filename}"?`)) {
        return;
    }
    
    try {
        const response = await fetch(`/documents/${documentId}`, {
            method: 'DELETE'
        });
        
        if (!response.ok) {
            const result = await response.json();
            throw new Error(result.error || 'Error deleting document');
        }
        
        // If document was part of the current chat, close the chat
        if (selectedDocuments.has(documentId)) {
            closeChat();
        }
        
        // Remove the document from selected documents if it was selected
        selectedDocuments.delete(documentId);
        
        // Reload the page to refresh the document list
        location.reload();
        
    } catch (error) {
        alert('Error deleting document: ' + error.message);
    }
}

// Add message to chat
function addMessage(role, content, sources = null) {
    const messagesDiv = document.getElementById('chatMessages');
    const messageDiv = document.createElement('div');
    messageDiv.className = `chat-message chat-message-${role}`;
    messageDiv.textContent = content;
    messagesDiv.appendChild(messageDiv);

    // Add sources if available
    if (sources && Array.isArray(sources) && sources.length > 0) {
        const sourcesDiv = document.createElement('div');
        sourcesDiv.className = 'chat-message chat-message-sources';
        sourcesDiv.innerHTML = '<strong>Sources:</strong><br>' + 
            sources.map(source => {
                if (typeof source === 'object' && source.content) {
                    return source.content;
                }
                return '';
            }).filter(content => content).join('<br><br>');
        
        if (sourcesDiv.innerHTML !== '<strong>Sources:</strong><br>') {
            messagesDiv.appendChild(sourcesDiv);
        }
    }

    messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

// Document ready handler
document.addEventListener('DOMContentLoaded', () => {
    // Start initialization
    initializeSystem();

    // Handle file upload
    document.getElementById('uploadForm').addEventListener('submit', async (e) => {
        e.preventDefault();
        const formData = new FormData(e.target);
        
        try {
            const response = await fetch('/documents/', {
                method: 'POST',
                body: formData
            });
            const result = await response.json();
            if (result.error) {
                alert(result.error);
            } else {
                location.reload();
            }
        } catch (error) {
            alert('Error uploading document');
        }
    });

    // Handle chat submission
    document.getElementById('chatForm').addEventListener('submit', async (e) => {
        e.preventDefault();
        if (selectedDocuments.size === 0) return;

        const messageInput = document.getElementById('messageInput');
        const message = messageInput.value.trim();
        if (!message) return;
        
        messageInput.value = '';
        
        // Disable input while processing
        messageInput.disabled = true;
        const submitButton = document.querySelector('#chatForm button[type="submit"]');
        submitButton.disabled = true;

        // Add user message to chat
        addMessage('user', message);
        addMessage('system', 'Processing your request...');

        try {
            // First check if the service is initialized
            const statusResponse = await fetch('/status');
            const statusData = await statusResponse.json();
            
            if (!statusData.is_initialized) {
                throw new Error(`Service is not ready yet. Status: ${statusData.status}`);
            }

            const response = await fetch(`/documents/query`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ 
                    query: message,
                    document_ids: Array.from(selectedDocuments)
                })
            });
            
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Error processing request');
            }
            
            const result = await response.json();
            
            // Remove the "Processing..." message
            document.getElementById('chatMessages').lastElementChild.remove();
            
            // Add the response with sources
            addMessage('assistant', result.response, result.sources);
            
        } catch (error) {
            // Remove the "Processing..." message
            document.getElementById('chatMessages').lastElementChild.remove();
            
            console.error('Chat error:', error);
            addMessage('error', `Error: ${error.message}`);
        } finally {
            // Re-enable input
            messageInput.disabled = false;
            submitButton.disabled = false;
        }
    });
}); 