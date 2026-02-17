import { Cosmograph, prepareCosmographData } from '@cosmograph/cosmograph';

const FC_COLORS = {
    Thing:   '#4FC3F7',
    Actor:   '#FF8A65',
    Place:   '#81C784',
    Event:   '#BA68C8',
    Concept: '#FFD54F',
    Time:    '#F06292',
};
const DEFAULT_COLOR = '#888888';

const datasetSelect = document.getElementById('dataset-select');
const datasetStatus = document.getElementById('dataset-status');
const loadingOverlay = document.getElementById('loading-overlay');
const truncationWarning = document.getElementById('truncation-warning');
const truncationMessage = document.getElementById('truncation-message');
const fcLegend = document.getElementById('fc-legend');
const graphStats = document.getElementById('graph-stats');
const graphContainer = document.getElementById('graph-container');

let currentDatasetId = null;
let cosmograph = null;

// Keep raw data for click lookups
let currentNodes = [];
let currentEdges = [];

// --- Helpers ---

function hideDetail() {
    const el = document.getElementById('node-detail');
    if (el) el.classList.add('d-none');
}

// --- Dataset loading (same pattern as chat.js) ---

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

function getEdgeType() {
    return document.querySelector('input[name="edge-type"]:checked').value;
}

async function loadGraph() {
    if (!currentDatasetId) return;

    loadingOverlay.classList.remove('d-none');
    truncationWarning.classList.add('d-none');
    hideDetail();

    const edgeType = getEdgeType();

    try {
        const resp = await fetch(`/api/graph/data?dataset_id=${currentDatasetId}&edge_type=${edgeType}`);
        if (!resp.ok) {
            console.error('Graph data error:', await resp.text());
            return;
        }
        const data = await resp.json();
        await renderGraph(data);
    } catch (err) {
        console.error('Error loading graph:', err);
    } finally {
        loadingOverlay.classList.add('d-none');
    }
}

async function renderGraph(data) {
    const { nodes, edges, metadata } = data;

    // Store for click lookups
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

    // Enrich nodes with color and size fields
    const rawPoints = nodes.map(n => ({
        id: n.id,
        label: n.label,
        color: FC_COLORS[n.fc] || DEFAULT_COLOR,
        size: n.pagerank > 0 ? 3 + 15 * Math.sqrt(n.pagerank / prMax) : 3,
    }));

    const rawLinks = edges.map(e => ({
        source: e.source,
        target: e.target,
    }));

    // Prepare data via Cosmograph's data pipeline
    const dataConfig = {
        points: {
            pointIdBy: 'id',
            pointLabelBy: 'label',
            pointColorBy: 'color',
            pointSizeBy: 'size',
        },
        links: {
            linkSourceBy: 'source',
            linkTargetsBy: ['target'],
        },
    };

    const result = await prepareCosmographData(dataConfig, rawPoints, rawLinks);
    if (!result) return;

    const { points, links, cosmographConfig } = result;

    // Destroy previous instance
    if (cosmograph) {
        graphContainer.innerHTML = '';
        cosmograph = null;
    }

    // Build edge index for click lookups (node id -> edges)
    const edgeIndex = new Map();
    for (const e of edges) {
        if (!edgeIndex.has(e.source)) edgeIndex.set(e.source, []);
        edgeIndex.get(e.source).push(e);
        if (!edgeIndex.has(e.target)) edgeIndex.set(e.target, []);
        edgeIndex.get(e.target).push(e);
    }

    cosmograph = new Cosmograph(graphContainer, {
        points,
        links,
        ...cosmographConfig,
        backgroundColor: '#1a1a2e',

        // Sizing: use computed sizes directly
        pointSizeStrategy: 'direct',
        pointSizeScale: 1,

        // Color: use hex colors directly
        pointColorStrategy: 'direct',

        // Links
        renderLinks: true,
        linkDefaultWidth: 2,
        linkOpacity: 0.6,
        linkGreyoutOpacity: 0.08,
        curvedLinks: true,
        curvedLinkWeight: 0.5,
        scaleLinksOnZoom: true,

        // Simulation
        simulationGravity: 0.15,
        simulationRepulsion: 0.8,

        // Labels
        showDynamicLabels: true,
        showTopLabels: true,
        showTopLabelsLimit: 20,
        pointLabelColor: '#ffffff',
        showHoveredPointLabel: true,
        showFocusedPointLabel: true,

        // Interaction
        selectPointOnClick: 'single',
        focusPointOnClick: true,
        renderHoveredPointRing: true,
        hoveredPointRingColor: '#ffffff',
        hoveredPointCursor: 'pointer',
        pointGreyoutOpacity: 0.15,

        // Click handler: show detail panel
        onPointClick: (clickedIndex) => {
            if (clickedIndex == null || clickedIndex < 0 || clickedIndex >= nodes.length) {
                hideDetail();
                return;
            }
            const node = nodes[clickedIndex];
            showNodeDetail(node, edgeIndex.get(node.id) || []);
        },

        onBackgroundClick: () => hideDetail(),
    });

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
    const fcColor = FC_COLORS[fc] || DEFAULT_COLOR;

    let html = '<dl>';

    // FC badge
    html += '<dt>Category</dt>';
    html += `<dd><span class="fc-badge" style="background:${fcColor};color:#000">${fc}</span></dd>`;

    // Type
    if (node.doc_type) {
        html += '<dt>Type</dt>';
        html += `<dd>${node.doc_type}</dd>`;
    }

    // PageRank
    if (node.pagerank > 0) {
        html += '<dt>PageRank</dt>';
        html += `<dd>${node.pagerank.toExponential(3)}</dd>`;
    }

    // URI
    html += '<dt>URI</dt>';
    html += `<dd>${node.id}</dd>`;

    // Connected edges
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
            `<span class="fc-legend-dot" style="background:${FC_COLORS[fc]}"></span>` +
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

document.querySelectorAll('input[name="edge-type"]').forEach(radio => {
    radio.addEventListener('change', () => loadGraph());
});

// Close button — use event delegation on document since the element
// lives outside the graph container and is always in the DOM
document.addEventListener('click', (e) => {
    if (e.target.id === 'node-detail-close' || e.target.closest('#node-detail-close')) {
        hideDetail();
    }
});

// Boot
loadDatasets();
