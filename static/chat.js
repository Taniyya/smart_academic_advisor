document.addEventListener('DOMContentLoaded', () => {
    const input = document.getElementById('chat-input');
    const sendBtn = document.getElementById('send-btn');
    const messagesContainer = document.getElementById('chat-messages');
    const typingIndicator = document.getElementById('typing-indicator');

    let chatHistory = [];

    function appendMessage(sender, text) {
        const msgDiv = document.createElement('div');
        msgDiv.className = `message ${sender}`;
        
        if (sender === 'ai') {
            msgDiv.innerHTML = marked.parse(text); // Use marked to render markdown
        } else {
            msgDiv.textContent = text;
        }
        
        messagesContainer.appendChild(msgDiv);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }

    async function sendMessage() {
        const text = input.value.trim();
        if (!text) return;

        // Display user user message
        appendMessage('user', text);
        input.value = '';
        input.disabled = true;
        sendBtn.disabled = true;
        typingIndicator.style.display = 'block';
        messagesContainer.scrollTop = messagesContainer.scrollHeight;

        try {
            const response = await fetch(`/api/chat/${studentId}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    message: text,
                    history: chatHistory
                })
            });

            const data = await response.json();
            typingIndicator.style.display = 'none';

            if (data.error) {
                appendMessage('ai', `**Error:** ${data.error}`);
            } else {
                appendMessage('ai', data.response || "Something went wrong.");
                // updating history
                chatHistory.push({ sender: 'user', text: text });
                chatHistory.push({ sender: 'ai', text: data.response });
            }

        } catch (error) {
            console.error(error);
            typingIndicator.style.display = 'none';
            appendMessage('ai', `**Error:** Failed to connect to server.`);
        } finally {
            input.disabled = false;
            sendBtn.disabled = false;
            input.focus();
        }
    }

    sendBtn.addEventListener('click', sendMessage);
    input.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') sendMessage();
    });
});
