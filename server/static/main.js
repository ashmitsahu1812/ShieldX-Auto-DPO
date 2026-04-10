document.addEventListener('DOMContentLoaded', () => {
    const dataBuffer = document.getElementById('data-buffer');
    const policyContext = document.getElementById('policy-context');
    const auditHistory = document.getElementById('audit-history');
    const executeBtn = document.getElementById('execute-btn');
    const resetBtn = document.getElementById('reset-btn');
    const taskSelect = document.getElementById('task-select');
    const scoreDisplay = document.getElementById('score-display');
    const stepCounter = document.getElementById('step-counter');
    const actionStatus = document.getElementById('action-status');
    const actionTarget = document.getElementById('action-target');
    const envInstruction = document.getElementById('env-instruction');

    let currentScore = 0.0;
    let stepCount = 0;

    async function initializeEnvironment() {
        const taskId = taskSelect.value;
        try {
            const response = await fetch(`/reset?task_id=${taskId}`, { method: 'POST' });
            const data = await response.json();
            updateState(data);
            auditHistory.innerHTML = '';
            showStatus('Environment Initialized', 'success');
            envInstruction.innerText = "System online. Awaiting compliance actions.";
        } catch (err) {
            showStatus('Failed to connect to server', 'error');
        }
    }

    async function fetchState() {
        try {
            const response = await fetch('/state');
            const data = await response.json();
            updateState(data);
        } catch (err) {
            console.error('State sync error', err);
        }
    }

    function updateState(state) {
        if (!state) return;
        
        // Update Viewports
        dataBuffer.innerText = state.data_buffer || 'No data in buffer.';
        policyContext.innerText = state.policy_context || 'No policy context active.';
        
        // Update Stats
        scoreDisplay.innerText = `Score: ${currentScore.toFixed(2)}`;
        stepCount = state.step || 0;
        stepCounter.innerText = `Step: ${stepCount}/5`;
        
        document.getElementById('task-badge').innerText = state.task_id || 'ShieldX';
    }

    async function executeAction() {
        const operation = document.querySelector('input[name="operation"]:checked').value;
        const target = actionTarget.value;

        if (!target) {
            showStatus('Target field is required', 'error');
            return;
        }

        executeBtn.disabled = true;
        executeBtn.innerText = 'EXECUTING...';

        try {
            const response = await fetch('/step', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ operation, target })
            });
            const result = await response.json();
            
            currentScore += result.reward;
            updateState(result.observation);
            
            addHistoryItem(operation, target, result.reward, result.info?.message);
            
            if (result.done) {
                showStatus(`Audit Session Complete! Final Score: ${currentScore.toFixed(2)}`, 'success');
                envInstruction.innerText = "Session terminated. Please reset to start a new audit.";
            } else {
                showStatus(`Action Success (+${result.reward.toFixed(2)})`, 'success');
            }

        } catch (err) {
            showStatus('Execution failed', 'error');
        } finally {
            executeBtn.disabled = false;
            executeBtn.innerText = 'EXECUTE ACTION';
            actionTarget.value = '';
        }
    }

    function addHistoryItem(op, target, reward, msg) {
        const item = document.createElement('div');
        item.className = 'history-item';
        item.innerHTML = `
            <div><span class="action">${op.toUpperCase()}</span> <span>"${target}"</span></div>
            <div class="ts">${new Date().toLocaleTimeString()}</div>
            <div class="reward">+${reward.toFixed(2)}</div>
        `;
        auditHistory.prepend(item);
    }

    function showStatus(msg, type) {
        actionStatus.innerText = msg;
        actionStatus.className = `action-alert alert-${type}`;
        actionStatus.classList.remove('hidden');
        setTimeout(() => actionStatus.classList.add('hidden'), 3000);
    }

    executeBtn.addEventListener('click', executeAction);
    resetBtn.addEventListener('click', initializeEnvironment);
    
    // Initial Load
    initializeEnvironment();
});
