import { Graph } from 'https://cdn.jsdelivr.net/npm/@cosmograph/cosmos@1.6.2-beta.1/+esm';

const FC_COLORS = {
    Thing:   [79, 195, 247, 1],   // #4FC3F7
    Actor:   [255, 138, 101, 1],  // #FF8A65
    Place:   [129, 199, 132, 1],  // #81C784
    Event:   [186, 104, 200, 1],  // #BA68C8
    Concept: [255, 213, 79, 1],   // #FFD54F
    Time:    [240, 98, 146, 1],   // #F06292
};
const FC_HEX = {
    Thing: '#4FC3F7', Actor: '#FF8A65', Place: '#81C784',
    Event: '#BA68C8', Concept: '#FFD54F', Time: '#F06292',
};
const DEFAULT_COLOR = [136, 136, 136, 1];
const DEFAULT_HEX = '#888888';

const datasetSelect = document.getElementById('dataset-select');
const datasetStatus = document.getElementById('dataset-status');
const loadingOverlay = document.getElementById('loading-overlay');
const truncationWarning = document.getElementById('truncation-warning');
const truncationMessage = document.getElementById('truncation-message');
const fcLegend = document.getElementById('fc-legend');
const graphStats = document.getElementById('graph-stats');
const canvas = document.getElementById('graph-canvas');

let currentDatasetId = null;
let graph = null;

// Keep raw data for click lookups
let currentNodes = [];
let currentEdges = [];
let nodeSizes = [];

// --- Helpers ---

function hideDetail() {
    const el = document.getElementById('node-detail');
    if (el) el.classList.add('d-none');
}

// Resize canvas to fill its wrapper
function resizeCanvas() {
    const wrapper = canvas.parentElement;
    canvas.width = wrapper.clientWidth;
    canvas.height = wrapper.clientHeight;
}

// --- Dataset loading ---

async function loadDatasets() {
    try {
        const response = await fetch('/api/datasets');
        if (!response.ok) return;
        const data = await response.json();

        datasetSelect.innerHTML = '<option value="" disabled selected>Select Dataset...</option>';
        data.datasets.forEach(ds => {
            const option = document.createElement('option');
            option.value = ds.id;
            let text = ds.display_name;
            if (!ds.has_cache) text += ' (not built)';
            else if (!ds.initialized) text += ' (not loaded)';
            option.textContent = text;
            datasetSelect.appendChild(option);
        });

        if (data.default) {
            datasetSelect.value = data.default;
            await selectDataset(data.default);
        }
    } catch (err) {
        console.error('Error loading datasets:', err);
        datasetStatus.textContent = 'Error';
        datasetStatus.className = 'badge bg-danger';
    }
}

async function selectDataset(datasetId) {
    if (!datasetId) return;

    datasetStatus.textContent = 'Loading...';
    datasetStatus.className = 'badge bg-warning';

    try {
        const resp = await fetch(`/api/datasets/${datasetId}/select`, { method: 'POST' });
        if (resp.ok) {
            currentDatasetId = datasetId;
            datasetStatus.textContent = 'Active';
            datasetStatus.className = 'badge bg-success';
            await loadGraph();
        } else {
            const err = await resp.json();
            datasetStatus.textContent = 'Error';
            datasetStatus.className = 'badge bg-danger';
            console.error('Dataset select error:', err);
        }
    } catch (err) {
        datasetStatus.textContent = 'Error';
        datasetStatus.className = 'badge bg-danger';
        console.error('Dataset select error:', err);
    }
}

// --- Graph loading & rendering ---

async function loadGraph() {
    if (!currentDatasetId) return;

    loadingOverlay.classList.remove('d-none');
    truncationWarning.classList.add('d-none');
    hideDetail();

    try {
        const resp = await fetch(`/api/graph/data?dataset_id=${currentDatasetId}`);
        if (!resp.ok) {
            console.error('Graph data error:', await resp.text());
            return;
        }
        const data = await resp.json();
        renderGraph(data);
    } catch (err) {
        console.error('Error loading graph:', err);
    } finally {
        loadingOverlay.classList.add('d-none');
    }
}

function renderGraph(data) {
    const { nodes, edges, metadata } = data;

    currentNodes = nodes;
    currentEdges = edges;

    // Truncation warning
    if (metadata.truncated) {
        truncationMessage.textContent =
            `Graph truncated: showing ${edges.length.toLocaleString()} of ${metadata.edge_count.toLocaleString()} edges.`;
        truncationWarning.classList.remove('d-none');
    }

    // Compute PageRank range for node sizing
    let prMax = 0;
    for (const n of nodes) {
        if (n.pagerank > prMax) prMax = n.pagerank;
    }
    if (prMax === 0) prMax = 1;

    // Pre-compute sizes
    nodeSizes = nodes.map(n =>
        n.pagerank > 0 ? 3 + 15 * Math.sqrt(n.pagerank / prMax) : 3
    );

    // Build cosmos input nodes/links
    const cosmosNodes = nodes.map(n => ({ id: n.id }));
    const cosmosLinks = edges.map(e => ({ source: e.source, target: e.target }));

    // Build edge index for click lookups
    const edgeIndex = new Map();
    for (const e of edges) {
        if (!edgeIndex.has(e.source)) edgeIndex.set(e.source, []);
        edgeIndex.get(e.source).push(e);
        if (!edgeIndex.has(e.target)) edgeIndex.set(e.target, []);
        edgeIndex.get(e.target).push(e);
    }

    // Build node index for fast lookup by id
    const nodeById = new Map();
    nodes.forEach((n, i) => nodeById.set(n.id, i));

    resizeCanvas();

    // Destroy previous instance
    if (graph) {
        graph.destroy();
        graph = null;
    }

    graph = new Graph(canvas, {
        backgroundColor: '#1a1a2e',

        // Node appearance
        nodeColor: (n, i) => FC_COLORS[nodes[i]?.fc] || DEFAULT_COLOR,
        nodeSize: (n, i) => nodeSizes[i] || 3,

        // Links
        renderLinks: true,
        linkWidth: 0.3,
        linkColor: [50, 50, 80, 0.6],
        curvedLinks: true,
        curvedLinkWeight: 0.5,

        // Simulation
        simulationGravity: 0.15,
        simulationRepulsion: 0.8,

        // Labels
        showDynamicLabels: true,
        showTopLabels: true,
        showTopLabelsLimit: 20,
        nodeLabelAccessor: (n, i) => nodes[i]?.label || '',
        nodeLabelColor: '#ffffff',

        // Interaction
        hoveredNodeRingColor: '#ffffff',

        onClick: (clickedNode, index, position, event) => {
            if (clickedNode == null || index == null || index < 0 || index >= nodes.length) {
                hideDetail();
                return;
            }
            const node = nodes[index];
            showNodeDetail(node, edgeIndex.get(node.id) || []);
        },
    });

    graph.setData(cosmosNodes, cosmosLinks);

    // Update legend & stats
    renderLegend(metadata.fc_counts);
    renderStats(metadata);
}

function showNodeDetail(node, nodeEdges) {
    const panel = document.getElementById('node-detail');
    const labelEl = document.getElementById('node-detail-label');
    const bodyEl = document.getElementById('node-detail-body');
    if (!panel || !labelEl || !bodyEl) return;

    labelEl.textContent = node.label;

    const fc = node.fc || 'Unknown';
    const fcColor = FC_HEX[fc] || DEFAULT_HEX;

    let html = '<dl>';

    html += '<dt>Category</dt>';
    html += `<dd><span class="fc-badge" style="background:${fcColor};color:#000">${fc}</span></dd>`;

    if (node.doc_type) {
        html += '<dt>Type</dt>';
        html += `<dd>${node.doc_type}</dd>`;
    }

    if (node.pagerank > 0) {
        html += '<dt>PageRank</dt>';
        html += `<dd>${node.pagerank.toExponential(3)}</dd>`;
    }

    html += '<dt>URI</dt>';
    html += `<dd>${node.id}</dd>`;

    if (nodeEdges.length > 0) {
        html += `<dt>Connections (${nodeEdges.length})</dt><dd>`;
        html += '<ul class="edge-list">';
        const shown = nodeEdges.slice(0, 30);
        for (const e of shown) {
            const isSource = e.source === node.id;
            const otherLabel = isSource
                ? (currentNodes.find(n => n.id === e.target)?.label || e.target.split('/').pop())
                : (currentNodes.find(n => n.id === e.source)?.label || e.source.split('/').pop());
            const arrow = isSource ? '&rarr;' : '&larr;';
            html += `<li>${arrow} ${otherLabel} <span class="edge-label">${e.predicate_label || e.edge_type}</span></li>`;
        }
        if (nodeEdges.length > 30) {
            html += `<li class="edge-label">... and ${nodeEdges.length - 30} more</li>`;
        }
        html += '</ul></dd>';
    }

    html += '</dl>';
    bodyEl.innerHTML = html;
    panel.classList.remove('d-none');
}

function renderLegend(fcCounts) {
    fcLegend.innerHTML = '';
    const allFcs = ['Thing', 'Actor', 'Place', 'Event', 'Concept', 'Time'];
    for (const fc of allFcs) {
        const count = fcCounts[fc] || 0;
        const item = document.createElement('div');
        item.className = 'fc-legend-item';
        item.innerHTML =
            `<span class="fc-legend-dot" style="background:${FC_HEX[fc]}"></span>` +
            `<span>${fc}</span>` +
            `<span class="fc-legend-count">(${count.toLocaleString()})</span>`;
        fcLegend.appendChild(item);
    }
}

function renderStats(meta) {
    graphStats.innerHTML = '';
    const items = [
        { label: 'Nodes', value: meta.node_count },
        { label: 'Edges', value: meta.edge_count },
    ];
    for (const s of items) {
        const div = document.createElement('div');
        div.className = 'stat-item';
        div.innerHTML =
            `<div class="stat-value">${s.value.toLocaleString()}</div>` +
            `<div class="stat-label">${s.label}</div>`;
        graphStats.appendChild(div);
    }
}

// --- Event listeners ---

datasetSelect.addEventListener('change', (e) => selectDataset(e.target.value));

document.addEventListener('click', (e) => {
    if (e.target.id === 'node-detail-close' || e.target.closest('#node-detail-close')) {
        hideDetail();
    }
});

window.addEventListener('resize', () => {
    resizeCanvas();
});

// Boot
loadDatasets();
