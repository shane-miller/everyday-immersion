/**
 * ChatApp - Main JavaScript class for the language learning chat interface
 * 
 * This class manages the interactive chat functionality, including message handling,
 * model status monitoring, and user interface interactions for the language learning app.
 */
class ChatApp {
    constructor() {
        // Initialize DOM element references
        this.chatMessages = document.getElementById('chatMessages');
        this.messageInput = document.getElementById('messageInput');
        this.sendButton = document.getElementById('sendButton');
        this.chatForm = document.getElementById('chatForm');
        this.charCount = document.getElementById('charCount');
        this.modelStatus = document.getElementById('modelStatus');
        this.preferredLanguageSelect = document.getElementById('preferredLanguageSelect');
        this.learningLanguageSelect = document.getElementById('learningLanguageSelect');
        this.beginSessionContainer = document.getElementById('beginSessionContainer');
        this.beginSessionButton = document.getElementById('beginSessionButton');
        this.modelSelect = document.getElementById('modelSelect');
        this.loadModelButton = document.getElementById('loadModelButton');
        this.resetButton = document.getElementById('resetButton');
        
        // Application state variables
        this.isLoading = false;
        this.modelReady = false;
        this.sessionStarted = false;
        this.sessionId = this.generateSessionId();
        this.availableModels = {};
        this.currentModel = null;
        
        this.init();
    }
    
    init() {
        // Initialize the chat application
        this.setupEventListeners();
        this.loadAvailableModels();
        this.autoResizeTextarea();
        this.updateBeginSessionButtonState();
        
        // Hide chat interface until session begins
        this.chatForm.parentElement.style.display = 'none';
        this.chatMessages.style.display = 'none';
    }
    
    setupEventListeners() {
        // Begin session button click handler
        this.beginSessionButton.addEventListener('click', () => {
            this.beginSession();
        });
        
        // Model selection handlers
        this.modelSelect.addEventListener('change', () => {
            this.updateLoadButtonState();
        });
        
        this.loadModelButton.addEventListener('click', () => {
            this.loadSelectedModel();
        });
        
        this.resetButton.addEventListener('click', () => {
            this.resetSession();
        });
        
        // Form submission handler
        this.chatForm.addEventListener('submit', (e) => {
            e.preventDefault();
            this.sendMessage();
        });
        
        // Character count and textarea resizing
        this.messageInput.addEventListener('input', () => {
            this.updateCharCount();
            this.autoResizeTextarea();
        });
        
        // Enter key handling for message submission
        this.messageInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });
        
        // Language selection change handlers
        this.preferredLanguageSelect.addEventListener('change', () => {
            this.updateLanguageContext();
            this.updateBeginSessionButtonState();
        });
        
        this.learningLanguageSelect.addEventListener('change', () => {
            this.updateLanguageContext();
            this.updateBeginSessionButtonState();
        });
    }
    
    async loadAvailableModels() {
        /**
         * Load available models from the server and populate the dropdown.
         */
        try {
            const response = await fetch('/models');
            const data = await response.json();
            
            this.availableModels = data.models;
            this.currentModel = data.current_model;
            
            this.populateModelDropdown();
            this.updateModelStatus();
        } catch (error) {
            console.error('Error loading models:', error);
            this.updateModelStatus('error', 'Failed to load models');
        }
    }
    
    populateModelDropdown() {
        /**
         * Populate the model selection dropdown with available models.
         */
        this.modelSelect.innerHTML = '<option value="">Choose a model...</option>';
        
        Object.entries(this.availableModels).forEach(([key, model]) => {
            const option = document.createElement('option');
            option.value = key;
            option.textContent = model.name;
            option.title = model.description;
            this.modelSelect.appendChild(option);
        });
        
        // Set current model if one is loaded
        if (this.currentModel) {
            this.modelSelect.value = this.currentModel;
            this.modelReady = true;
        }
        
        this.updateLoadButtonState();
        this.updateBeginSessionButtonState();
    }
    
    updateLoadButtonState() {
        /**
         * Update the load button state based on model selection.
         */
        const selectedModel = this.modelSelect.value;
        this.loadModelButton.disabled = !selectedModel || this.isLoading;
    }
    
    updateBeginSessionButtonState() {
        /**
         * Update the begin session button state based on model readiness and language selection.
         */
        const preferredLanguage = this.preferredLanguageSelect.value;
        const learningLanguage = this.learningLanguageSelect.value;
        const languagesSelected = preferredLanguage && learningLanguage;
        
        this.beginSessionButton.disabled = !this.modelReady || !languagesSelected || this.isLoading;
    }
    
    async loadSelectedModel() {
        /**
         * Load the selected model from the dropdown.
         */
        const selectedModel = this.modelSelect.value;
        if (!selectedModel) return;
        
        this.setLoading(true);
        this.updateModelStatus('loading', 'Loading model...');
        
        try {
            const response = await fetch('/load-model', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    model_key: selectedModel
                })
            });
            
            const data = await response.json();
            
            if (data.status === 'success') {
                this.modelReady = true;
                this.currentModel = selectedModel;
                this.updateModelStatus('ready', `Model ${this.availableModels[selectedModel].name} ready`);
                this.updateBeginSessionButtonState();
            } else {
                throw new Error(data.error || 'Failed to load model');
            }
        } catch (error) {
            console.error('Error loading model:', error);
            this.updateModelStatus('error', `Failed to load model: ${error.message}`);
        } finally {
            this.setLoading(false);
            this.updateLoadButtonState();
        }
    }
    
    updateModelStatus(status = null, message = null) {
        /**
         * Update the visual model status indicator in the UI.
         * 
         * Args:
         *     status (string): Status type ('ready', 'loading', 'error', 'not-loaded')
         *     message (string): Status message to display
         */
        const statusIndicator = this.modelStatus.querySelector('.status-indicator');
        
        if (status) {
            statusIndicator.className = `status-indicator ${status}`;
        }
        
        const icon = statusIndicator.querySelector('i');
        if (status === 'ready') {
            icon.className = 'fas fa-check';
        } else if (status === 'error') {
            icon.className = 'fas fa-exclamation-triangle';
        } else if (status === 'loading') {
            icon.className = 'fas fa-spinner fa-spin';
        } else if (status === 'not-loaded') {
            icon.className = 'fas fa-info-circle';
        }
        
        if (message) {
            statusIndicator.textContent = message;
            statusIndicator.appendChild(icon);
        }
    }
    
    async beginSession() {
        /**
         * Initialize the language learning session.
         * 
         * This method transitions from the welcome screen to the active chat
         * interface and sends an initial message to start the learning session.
         */
        if (this.isLoading || !this.modelReady) return;
        
        // Hide the welcome screen
        this.beginSessionContainer.style.display = 'none';
        
        // Show the chat interface
        this.chatForm.parentElement.style.display = 'block';
        this.chatMessages.style.display = 'block';
        
        // Show the reset button
        this.resetButton.style.display = 'flex';
        
        // Send initial session start message with language context
        const languageContext = `Begin learning session. I speak ${this.preferredLanguageSelect.value} and want to learn ${this.learningLanguageSelect.value}.`;
        await this.sendMessage(languageContext, true);
        
        this.sessionStarted = true;
    }
    
    async sendMessage(message = null, isInitialMessage = false) {
        /**
         * Send a message to the language learning AI and handle the response.
         * 
         * Args:
         *     message (string, optional): Message text to send
         *     isInitialMessage (boolean): Whether this is the session start message
         */
        const messageText = message || this.messageInput.value.trim();
        if (!messageText || this.isLoading) return;
        
        // Add user message to chat display
        this.addMessage(messageText, 'user');
        
        if (!isInitialMessage) {
            this.messageInput.value = '';
            this.updateCharCount();
            this.autoResizeTextarea();
        }
        
        // Show typing indicator and disable input during processing
        this.showTypingIndicator();
        this.setLoading(true);
        
        try {
            // Send message to backend API
            const response = await fetch('/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    message: messageText,
                    sessionId: this.sessionId
                })
            });
            
            const data = await response.json();
            
            if (data.status === 'success') {
                this.hideTypingIndicator();
                this.addMessage(data.response, 'bot');
            } else {
                throw new Error(data.error || 'Failed to get response');
            }
        } catch (error) {
            console.error('Error sending message:', error);
            this.hideTypingIndicator();
            this.addMessage('Sorry, I encountered an error. Please try again.', 'bot');
        } finally {
            this.setLoading(false);
        }
    }
    
    addMessage(content, sender) {
        /**
         * Add a message to the chat display.
         * 
         * Args:
         *     content (string): Message text content
         *     sender (string): Message sender ('user' or 'bot')
         */
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}-message`;
        
        // Create avatar element
        const avatar = document.createElement('div');
        avatar.className = 'message-avatar';
        
        const icon = document.createElement('i');
        icon.className = sender === 'user' ? 'fas fa-user' : 'fas fa-robot';
        avatar.appendChild(icon);
        
        // Create message content container
        const messageContent = document.createElement('div');
        messageContent.className = 'message-content';
        
        const paragraph = document.createElement('p');
        paragraph.textContent = content;
        messageContent.appendChild(paragraph);
        
        // Assemble message element
        messageDiv.appendChild(avatar);
        messageDiv.appendChild(messageContent);
        
        this.chatMessages.appendChild(messageDiv);
        this.scrollToBottom();
    }
    
    showTypingIndicator() {
        /**
         * Display a typing indicator to show the AI is processing.
         * 
         * Creates an animated typing indicator with three dots
         * to provide visual feedback during response generation.
         */
        const typingDiv = document.createElement('div');
        typingDiv.className = 'message bot-message';
        typingDiv.id = 'typingIndicator';
        
        const avatar = document.createElement('div');
        avatar.className = 'message-avatar';
        
        const icon = document.createElement('i');
        icon.className = 'fas fa-robot';
        avatar.appendChild(icon);
        
        const messageContent = document.createElement('div');
        messageContent.className = 'message-content typing-indicator';
        
        // Create animated typing dots
        for (let i = 0; i < 3; i++) {
            const dot = document.createElement('div');
            dot.className = 'typing-dot';
            messageContent.appendChild(dot);
        }
        
        typingDiv.appendChild(avatar);
        typingDiv.appendChild(messageContent);
        
        this.chatMessages.appendChild(typingDiv);
        this.scrollToBottom();
    }
    
    hideTypingIndicator() {
        /**
         * Remove the typing indicator from the chat display.
         */
        const typingIndicator = document.getElementById('typingIndicator');
        if (typingIndicator) {
            typingIndicator.remove();
        }
    }
    
    setLoading(loading) {
        /**
         * Set the loading state and update UI accordingly.
         * 
         * Args:
         *     loading (boolean): Whether the app is in loading state
         */
        this.isLoading = loading;
        this.sendButton.disabled = loading;
        this.messageInput.disabled = loading;
        this.updateLoadButtonState();
        this.updateBeginSessionButtonState();
    }
    
    updateCharCount() {
        /**
         * Update the character count display with color coding.
         * 
         * Provides visual feedback on message length with color changes
         * as the user approaches the 500 character limit.
         */
        const count = this.messageInput.value.length;
        this.charCount.textContent = `${count}/500`;
        
        if (count > 450) {
            this.charCount.style.color = '#e53e3e';  // Red for near limit
        } else if (count > 400) {
            this.charCount.style.color = '#dd6b20';  // Orange for warning
        } else {
            this.charCount.style.color = '#718096';  // Gray for normal
        }
    }
    
    autoResizeTextarea() {
        /**
         * Automatically resize the message input textarea based on content.
         * 
         * Provides a better user experience by allowing the input field
         * to grow with the message content up to a maximum height.
         */
        this.messageInput.style.height = 'auto';
        this.messageInput.style.height = Math.min(this.messageInput.scrollHeight, 120) + 'px';
    }
    
    scrollToBottom() {
        /**
         * Scroll the chat messages container to the bottom.
         * 
         * Ensures the latest messages are always visible to the user.
         */
        this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
    }
    
    generateSessionId() {
        /**
         * Generate a unique session identifier.
         * 
         * Returns:
         *     string: Unique session ID
         */
        return 'session_' + Date.now() + '_' + Math.random().toString(36).substring(2, 11);
    }
    
    updateLanguageContext() {
        /**
         * Update the language learning context when language selections change.
         * 
         * Logs the current language configuration for debugging purposes.
         */
        const preferredLanguage = this.preferredLanguageSelect.value;
        const learningLanguage = this.learningLanguageSelect.value;
        
        console.log(`Language context: ${preferredLanguage} speaker learning ${learningLanguage}`);
    }
    
    async resetSession() {
        /**
         * Reset the current session and return to the initial state.
         * 
         * This method clears the conversation history, resets the session state,
         * and returns the user to the model and language selection screen.
         */
        if (this.isLoading) return;
        
        try {
            // Call the reset endpoint
            const response = await fetch('/reset', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    sessionId: this.sessionId
                })
            });
            
            const data = await response.json();
            
            if (data.status === 'success') {
                // Clear chat messages
                this.chatMessages.innerHTML = '';
                
                // Reset session state
                this.sessionStarted = false;
                this.sessionId = this.generateSessionId();
                
                // Reset model state since it was unloaded
                this.modelReady = false;
                this.currentModel = null;
                
                // Reset model selection dropdown
                this.modelSelect.value = '';
                
                // Update model status to show no model loaded
                this.updateModelStatus('not-loaded', 'No model loaded');
                
                // Show the welcome screen again
                this.beginSessionContainer.style.display = 'flex';
                
                // Hide the chat interface
                this.chatForm.parentElement.style.display = 'none';
                this.chatMessages.style.display = 'none';
                
                // Hide the reset button
                this.resetButton.style.display = 'none';
                
                // Update button states
                this.updateLoadButtonState();
                this.updateBeginSessionButtonState();
                
                console.log('Session reset successfully - model unloaded');
            } else {
                throw new Error(data.error || 'Failed to reset session');
            }
        } catch (error) {
            console.error('Error resetting session:', error);
            alert('Failed to reset session. Please try again.');
        }
    }
}

// Initialize the chat application when the DOM is fully loaded
document.addEventListener('DOMContentLoaded', () => {
    new ChatApp();
}); 