let currentDocumentId = null;

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
            // Check again in 2 seconds
            setTimeout(checkInitStatus, 2000);
        }
    } catch (error) {
        console.error('Error checking initialization status:', error);
    }
}

// Start initialization and status checking
async function initializeSystem() {
    try {
        await fetch('/initialize', { method: 'POST' });
        checkInitStatus();
    } catch (error) {
        console.error('Error starting initialization:', error);
    }
}

// Handle document selection
function selectDocument(documentId) {
    currentDocumentId = documentId;
    document.getElementById('chatSection').style.display = 'block';
    document.getElementById('chatMessages').innerHTML = '';
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

// Handle document summarization
async function summarizeDocument() {
    if (!currentDocumentId) return;

    addMessage('system', 'Generating document summary...');

    try {
        const response = await fetch(`/documents/${currentDocumentId}/summary`);
        const result = await response.json();
        
        if (result.error) {
            addMessage('error', result.error);
        } else {
            addMessage('system', 'Document Summary:');
            addMessage('assistant', result.summary);
        }
    } catch (error) {
        addMessage('error', 'Error generating summary');
    }
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
        if (!currentDocumentId) return;

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

            const response = await fetch(`/documents/${currentDocumentId}/query`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ query: message })
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