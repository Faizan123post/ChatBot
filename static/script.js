 // static/script.js
document.addEventListener('DOMContentLoaded', function() {
    const chatMessages = document.getElementById('chat-messages');
    const queryForm = document.getElementById('query-form');
    const queryInput = document.getElementById('query-input');
    const refreshButton = document.getElementById('refresh-docs');
    
    // Submit query when form is submitted
    queryForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        const query = queryInput.value.trim();
        if (!query) return;
        
        // Display user's message
        addMessage(query, 'user');
        
        // Clear input
        queryInput.value = '';
        
        // Show loading indicator
        const loadingMessageId = addLoadingMessage();
        
        // Send query to API
        fetch('/api/query', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ question: query })
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            // Remove loading message
            removeLoadingMessage(loadingMessageId);
            
            // Display bot's response
            addMessage(data.answer, 'system');
            
            // Scroll to bottom
            scrollToBottom();
        })
        .catch(error => {
            // Remove loading message
            removeLoadingMessage(loadingMessageId);
            
            // Display error message
            addMessage('Sorry, I had trouble processing your question. Please try again.', 'system');
            console.error('Error fetching response:', error);
            
            // Scroll to bottom
            scrollToBottom();
        });
    });
    
    // Refresh documentation when button is clicked
    refreshButton.addEventListener('click', function() {
        const originalText = refreshButton.textContent;
        refreshButton.textContent = 'Refreshing...';
        refreshButton.disabled = true;
        
        fetch('/api/refresh_docs')
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            addMessage('Documentation has been refreshed successfully!', 'system');
            scrollToBottom();
            refreshButton.textContent = originalText;
            refreshButton.disabled = false;
        })
        .catch(error => {
            addMessage('Failed to refresh documentation. Please try again later.', 'system');
            scrollToBottom();
            console.error('Error refreshing docs:', error);
            refreshButton.textContent = originalText;
            refreshButton.disabled = false;
        });
    });
    
    // Function to add a message to the chat
    function addMessage(content, type) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${type}`;
        
        const messageContent = document.createElement('div');
        messageContent.className = 'message-content';
        
        // Convert URLs to clickable links
        if (type === 'system') {
            // Process Markdown-style links and formatting
            content = processMarkdown(content);
        } else {
            // For user messages, just show the text
            messageContent.textContent = content;
        }
        
        if (type === 'system') {
            messageContent.innerHTML = content;
        } else {
            messageContent.textContent = content;
        }
        
        messageDiv.appendChild(messageContent);
        chatMessages.appendChild(messageDiv);
        
        scrollToBottom();
        return messageDiv.id;
    }
    
    // Function to add a loading message
    function addLoadingMessage() {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message system';
        messageDiv.id = 'loading-message-' + Date.now();
        
        const messageContent = document.createElement('div');
        messageContent.className = 'message-content';
        
        const loadingDots = document.createElement('div');
        loadingDots.className = 'loading-dots';
        loadingDots.innerHTML = '<span></span><span></span><span></span>';
        
        messageContent.appendChild(document.createTextNode('Thinking '));
        messageContent.appendChild(loadingDots);
        
        messageDiv.appendChild(messageContent);
        chatMessages.appendChild(messageDiv);
        
        scrollToBottom();
        return messageDiv.id;
    }
    
    // Function to remove a loading message
    function removeLoadingMessage(id) {
        const loadingMessage = document.getElementById(id);
        if (loadingMessage) {
            loadingMessage.remove();
        }
    }
    
    // Function to scroll chat to bottom
    function scrollToBottom() {
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
    
    // Function to process markdown-like syntax
    function processMarkdown(text) {
        // Convert URLs to links
        text = text.replace(
            /(\b(https?|ftp|file):\/\/[-A-Z0-9+&@#\/%?=~_|!:,.;]*[-A-Z0-9+&@#\/%=~_|])/ig,
            '<a href="$1" target="_blank">$1</a>'
        );
        
        // Process code blocks (not perfect but works for simple cases)
        text = text.replace(/```([\s\S]*?)```/g, '<pre><code>$1</code></pre>');
        
        // Process inline code
        text = text.replace(/`([^`]+)`/g, '<code>$1</code>');
        
        // Process bold
        text = text.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
        
        // Process italics
        text = text.replace(/\*([^*]+)\*/g, '<em>$1</em>');
        
        // Process headers (## Header)
        text = text.replace(/^## (.*$)/gm, '<h2>$1</h2>');
        
        // Process lists (simple bullet points)
        text = text.replace(/^\s*\*\s(.*$)/gm, '<li>$1</li>');
        text = text.replace(/<li>(.*)<\/li>/gm, function(match) {
            return '<ul>' + match + '</ul>';
        });
        
        // Convert newlines to <br> or <p>
        text = text.replace(/\n\n/g, '</p><p>');
        text = '<p>' + text + '</p>';
        text = text.replace(/<p><\/p>/g, '');
        
        return text;
    }
    
    // Focus input on page load
    queryInput.focus();
});