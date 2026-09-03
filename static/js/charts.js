/**
 * Smart Energy AI - Chart Rendering Logic (Plotly.js)
 */

document.addEventListener('DOMContentLoaded', () => {
    // Delay slightly to ensure UI is ready
    setTimeout(() => {
        loadHourlyChart();
        loadDailyChart();
        loadHeatmap();
        loadPerformanceChart();
    }, 500);
});

// Common Plotly layout styling for dark theme
const commonLayout = {
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    font: { color: '#9CA3AF', family: 'Inter, sans-serif' },
    margin: { t: 30, r: 20, l: 50, b: 40 },
    xaxis: { 
        gridcolor: '#374151', 
        zerolinecolor: '#374151' 
    },
    yaxis: { 
        gridcolor: '#374151', 
        zerolinecolor: '#374151',
        title: { text: 'Wh', font: { size: 11 } }
    },
    hovermode: 'closest',
    autosize: true
};

async function loadHourlyChart() {
    try {
        const res = await fetch('/api/analytics/hourly');
        const data = await res.json();
        
        if (data.error) {
            document.getElementById('chart-hourly').innerHTML = `<p class="text-secondary" style="padding: 20px;">${data.error}</p>`;
            return;
        }

        const trace = {
            x: data.hours.map(h => `${String(h).padStart(2, '0')}:00`),
            y: data.averages,
            type: 'scatter',
            mode: 'lines+markers',
            line: { color: '#10B981', width: 3, shape: 'spline' },
            marker: { size: 6, color: '#14B8A6' },
            name: 'Average Consumption'
        };

        const layout = {
            ...commonLayout,
            hoverlabel: { bgcolor: '#1F2937' },
            xaxis: { ...commonLayout.xaxis, tickangle: -45 }
        };

        Plotly.newPlot('chart-hourly', [trace], layout, { responsive: true, displayModeBar: false });
    } catch (err) {
        console.error("Failed to load hourly chart", err);
    }
}

async function loadDailyChart() {
    try {
        const res = await fetch('/api/analytics/daily');
        const data = await res.json();
        
        if (data.error) {
            document.getElementById('chart-daily').innerHTML = `<p class="text-secondary" style="padding: 20px;">${data.error}</p>`;
            return;
        }

        const trace = {
            x: data.days,
            y: data.averages,
            type: 'bar',
            marker: { 
                color: data.averages, 
                colorscale: 'Tealgrn', 
                reversescale: true 
            },
            name: 'Daily Average'
        };

        const layout = {
            ...commonLayout
        };

        Plotly.newPlot('chart-daily', [trace], layout, { responsive: true, displayModeBar: false });
    } catch (err) {
        console.error("Failed to load daily chart", err);
    }
}

async function loadHeatmap() {
    try {
        const response = await fetch('/api/analytics/heatmap');
        const data = await response.json();

        if (!response.ok || data.error) {
            document.getElementById('chart-heatmap').innerHTML = `<p class="text-secondary" style="padding: 20px;">Data unavailable.</p>`;
            return;
        }

        const trace = {
            z: data.values,
            x: data.hours.map(h => `${String(h).padStart(2, '0')}:00`),
            y: data.days,
            type: 'heatmap',
            colorscale: 'Tealgrn',
            reversescale: true,
            hoverongaps: false
        };

        const layout = {
            ...commonLayout,
            margin: { t: 20, r: 20, l: 80, b: 60 }
        };

        Plotly.newPlot('chart-heatmap', [trace], layout, { responsive: true });
    } catch (err) {
        console.error("Failed to load heatmap", err);
    }
}

async function loadPerformanceChart() {
    try {
        const res = await fetch('/api/model/performance');
        const data = await res.json();

        if (data.error) {
            document.getElementById('chart-performance').innerHTML = `<p class="text-secondary" style="padding: 20px;">${data.error}</p>`;
            return;
        }

        // Update KPI text
        document.getElementById('perf-mae').textContent = data.MAE;
        document.getElementById('perf-rmse').textContent = data.RMSE;
        document.getElementById('perf-r2').textContent = data.R2;
        document.getElementById('perf-mape').textContent = data.MAPE + ' %';

        // Render Scatter / Line comparison
        const traceActual = {
            y: data.actual_subset,
            type: 'scatter',
            mode: 'lines',
            name: 'Actual',
            line: { color: '#06B6D4' }
        };

        const tracePredicted = {
            y: data.predicted_subset,
            type: 'scatter',
            mode: 'lines',
            name: 'Predicted',
            line: { color: '#10B981', dash: 'dot' }
        };

        const layout = {
            ...commonLayout,
            xaxis: { ...commonLayout.xaxis, title: { text: 'Time (Test Samples Subset)' } },
            yaxis: { ...commonLayout.yaxis, title: { text: 'Wh' } }
        };

        Plotly.newPlot('chart-performance', [traceActual, tracePredicted], layout, { responsive: true });
    } catch (err) {
        console.error("Failed to load performance chart", err);
    }
}
