/**
 * Smart Energy AI - Main Application Logic
 */

document.addEventListener('DOMContentLoaded', () => {
    initNavigation();
    checkHealth();
    loadDashboardKPIs();
    loadSettingsInfo();
    setupPredictionForm();
});

// --- UI Components --- //

const ui = {
    modal: document.getElementById('global-modal'),
    modalTitle: document.getElementById('modal-title'),
    modalBody: document.getElementById('modal-body'),
    modalClose: document.getElementById('modal-close'),
    modalBtnPrimary: document.getElementById('modal-btn-primary'),
    toastContainer: document.getElementById('toast-container'),
    apiStatus: document.getElementById('api-status'),
    menuToggle: document.getElementById('menu-toggle'),
    sidebar: document.getElementById('sidebar')
};

// --- Navigation --- //

function initNavigation() {
    const navLinks = document.querySelectorAll('.nav-link');
    const sections = document.querySelectorAll('.content-section');
    const pageTitle = document.getElementById('page-title');

    navLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            // Update active link
            navLinks.forEach(l => l.classList.remove('active'));
            link.classList.add('active');
            
            // Update Title
            pageTitle.textContent = link.textContent.trim();

            // Toggle sections
            const targetId = link.getAttribute('data-target');
            sections.forEach(sec => {
                if (sec.id === targetId) {
                    sec.classList.add('active');
                    // Trigger resize for plotly charts if needed
                    if (window.dispatchEvent) {
                        window.dispatchEvent(new Event('resize'));
                    }
                } else {
                    sec.classList.remove('active');
                }
            });

            // Close sidebar on mobile
            if (window.innerWidth <= 768) {
                ui.sidebar.classList.remove('open');
            }
        });
    });

    ui.menuToggle.addEventListener('click', () => {
        ui.sidebar.classList.toggle('open');
    });
}

// --- API & State --- //

async function checkHealth() {
    try {
        const res = await fetch('/api/health');
        const data = await res.json();
        if (res.ok && data.status === 'ok') {
            ui.apiStatus.classList.add('connected');
            ui.apiStatus.title = "Connected to Backend";
        } else {
            ui.apiStatus.classList.add('error');
            ui.apiStatus.title = "Backend needs attention";
        }
    } catch (err) {
        ui.apiStatus.classList.add('error');
        ui.apiStatus.title = "Connection Failed";
        showToast("Cannot connect to backend server.", "error");
    }
}

async function loadSettingsInfo() {
    try {
        const res = await fetch('/api/info');
        if (res.ok) {
            const data = await res.json();
            document.getElementById('setting-model-type').textContent = data.model_type;
            document.getElementById('setting-threshold').textContent = data.threshold_wh;
        }
    } catch (err) {
        console.error("Failed to load settings info", err);
    }
}

async function loadDashboardKPIs() {
    try {
        const res = await fetch('/api/analytics/summary');
        if (res.ok) {
            const data = await res.json();
            if(!data.error) {
                document.getElementById('kpi-avg').textContent = data.average_consumption_wh + ' Wh';
                document.getElementById('kpi-latest').textContent = data.latest_consumption_wh + ' Wh';
                document.getElementById('kpi-peak').textContent = data.peak_hour;
            }
        }
    } catch (err) {
        console.error("Failed to load KPI summary", err);
    }
}

// --- Prediction Form --- //

function setupPredictionForm() {
    const form = document.getElementById('prediction-form');
    const btn = document.getElementById('btn-predict');

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        // Form Loading State
        const originalText = btn.innerHTML;
        btn.innerHTML = '<span>Processing...</span><i class="fa-solid fa-spinner fa-spin"></i>';
        btn.disabled = true;

        // Build Payload
        const formData = new FormData(form);
        const payload = {};
        for (let [key, value] of formData.entries()) {
            payload[key] = (key === 'selected_date' || key === 'selected_time') ? value : parseFloat(value);
        }

        try {
            const res = await fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            
            const data = await res.json();
            
            if (res.ok && data.success) {
                handlePredictionSuccess(data, payload);
            } else {
                showModal("Prediction Error", data.detail || "An unexpected error occurred during prediction.");
            }
        } catch (err) {
            showModal("Connection Error", "Failed to communicate with the prediction server.");
        } finally {
            // Restore button
            btn.innerHTML = originalText;
            btn.disabled = false;
        }
    });
}

function handlePredictionSuccess(data, inputData) {
    // 1. Update KPI
    const statusKpi = document.getElementById('kpi-status');
    statusKpi.textContent = data.level;
    statusKpi.className = 'kpi-value'; // reset
    if (data.level === 'High') {
        statusKpi.classList.add('text-cyan'); // Actually, let's use a red for high in CSS, but the prompt says use semantic. We'll add style directly or classes
        statusKpi.style.color = 'var(--status-high)';
    } else {
        statusKpi.style.color = 'var(--status-normal)';
    }

    // 2. Update AI Insights Section
    const insightContent = document.getElementById('insight-content');
    insightContent.className = ''; 
    let html = `
        <div style="font-size: 2rem; font-weight: bold; margin-bottom: 0.5rem; color: var(--text-primary);">
            ${data.predicted_wh} Wh
        </div>
        <div class="badge" style="background-color: ${data.level === 'High' ? 'rgba(239, 68, 68, 0.1)' : 'rgba(16, 185, 129, 0.1)'}; color: ${data.level === 'High' ? 'var(--status-high)' : 'var(--status-normal)'}; display: inline-block; margin-bottom: 1rem;">
            ${data.level} Consumption Level
        </div>
        <p>${data.message}</p>
    `;
    
    // Generate Recommendation based on data
    if (data.level === 'High') {
        html += `
            <div class="alert-box alert-warning mt-2">
                <i class="fa-solid fa-triangle-exclamation"></i>
                <div>
                    <strong>Recommendation:</strong> Predicted consumption is above the threshold of ${data.threshold_wh} Wh. 
                    Consider reducing simultaneous appliance usage or unnecessary lighting during this period.
                </div>
            </div>
        `;
    } else {
        html += `
            <div class="alert-box alert-info mt-2">
                <i class="fa-solid fa-leaf"></i>
                <div>
                    <strong>Efficiency Note:</strong> Expected consumption is within normal historical patterns for the selected conditions.
                </div>
            </div>
        `;
    }
    insightContent.innerHTML = html;

    // 3. Show Success Modal
    showModal(
        "Prediction Complete", 
        `The model expects household appliance consumption to be approximately <strong>${data.predicted_wh} Wh</strong> under the selected conditions.<br><br>Status: <strong>${data.level}</strong>`,
        () => {
            // Callback: navigate to insights
            document.querySelector('.nav-link[data-target="insights-section"]').click();
        }
    );
    
    showToast("Prediction successful", "success");
}

// --- Modals & Toasts --- //

function showModal(title, message, onOkCallback = null) {
    ui.modalTitle.textContent = title;
    ui.modalBody.innerHTML = message;
    ui.modal.classList.add('show');
    
    // Clear old listeners
    const newBtn = ui.modalBtnPrimary.cloneNode(true);
    ui.modalBtnPrimary.parentNode.replaceChild(newBtn, ui.modalBtnPrimary);
    ui.modalBtnPrimary = newBtn;
    
    const closeModal = () => ui.modal.classList.remove('show');
    
    ui.modalClose.onclick = closeModal;
    ui.modalBtnPrimary.onclick = () => {
        closeModal();
        if (onOkCallback) onOkCallback();
    };
}

function showToast(message, type = "info") {
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;
    ui.toastContainer.appendChild(toast);
    
    setTimeout(() => {
        if (toast.parentNode) {
            toast.parentNode.removeChild(toast);
        }
    }, 3500);
}
