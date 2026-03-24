(() => {
    const state = {
        bootstrap: null,
        map: null,
        tileLayer: null,
        geojsonLayer: null,
        lamahGeojsonLayer: null,
        markers: new Map(),
        stationsByName: new Map(),
        activeFeatureFilter: 'All',
        activeDatasetFilter: 'all',
        focusedStation: null,
        cardCounters: {
            visualize: 0,
            analysis: 0,
            prediction: 0,
        },
    };

    const els = {
        sidebar: document.getElementById('sidebar'),
        sidebarToggle: document.getElementById('sidebarToggle'),
        stationSearch: document.getElementById('stationSearch'),
        stationSuggestList: document.getElementById('stationSuggestList'),
        focusStationBtn: document.getElementById('focusStationBtn'),
        clearStationFocusBtn: document.getElementById('clearStationFocusBtn'),
        featureFilterBar: document.getElementById('featureFilterBar'),
        datasetFilterBar: document.getElementById('datasetFilterBar'),
        graphTypeSelect: document.getElementById('graphTypeSelect'),
        graphInfoBtn: document.getElementById('graphInfoBtn'),
        graphInfoPopover: document.getElementById('graphInfoPopover'),
        closePopoverBtn: document.getElementById('closePopoverBtn'),
        popoverTitle: document.getElementById('popoverTitle'),
        popoverDesc: document.getElementById('popoverDesc'),
        popoverRules: document.getElementById('popoverRules'),
        seriesBuilder: document.getElementById('seriesBuilder'),
        seriesRowTemplate: document.getElementById('seriesRowTemplate'),
        addSeriesRowBtn: document.getElementById('addSeriesRowBtn'),
        addVisualizationBtn: document.getElementById('addVisualizationBtn'),
        addAnalysisBtn: document.getElementById('addAnalysisBtn'),
        builderMessage: document.getElementById('builderMessage'),
        datasetPicker: document.getElementById('datasetPicker'),
        flyToMekong: document.getElementById('flyToMekong'),
        flyToLamaH: document.getElementById('flyToLamaH'),
        stationDetailCard: document.getElementById('stationDetailCard'),
        detailStationName: document.getElementById('detailStationName'),
        stationMetaGrid: document.getElementById('stationMetaGrid'),
        stationFeatureBadges: document.getElementById('stationFeatureBadges'),
        closeStationCardBtn: document.getElementById('closeStationCardBtn'),
        dock: document.getElementById('dock'),
        dockHandle: document.getElementById('dockHandle'),
        dockTabs: Array.from(document.querySelectorAll('.dock-tab')),
        dockPanels: Array.from(document.querySelectorAll('.dock-panel')),
        workspacePills: Array.from(document.querySelectorAll('.workspace-pill')),
        dockClearBtns: Array.from(document.querySelectorAll('.dock-clear-btn')),
        visualizationCards: document.getElementById('visualizationCards'),
        analysisCards: document.getElementById('analysisCards'),
        predictionCards: document.getElementById('predictionCards'),
        clearVisualizationsBtn: document.getElementById('clearVisualizationsBtn'),
        clearAnalysisBtn: document.getElementById('clearAnalysisBtn'),
        clearPredictionBtn: document.getElementById('clearPredictionBtn'),
        predictStationSelect: document.getElementById('predictStationSelect'),
        predictFeatureSelect: document.getElementById('predictFeatureSelect'),
        predictHorizonInput: document.getElementById('predictHorizonInput'),
        predictHint: document.getElementById('predictHint'),
        runPredictionBtn: document.getElementById('runPredictionBtn'),
        predictionMessage: document.getElementById('predictionMessage'),
        seriesRowTemplate: document.getElementById('seriesRowTemplate'),
        featureFirstTemplate: document.getElementById('featureFirstTemplate'),
        stationRowTemplate: document.getElementById('stationRowTemplate'),
        mapSearchBar: document.getElementById('mapSearchBar'),
        mapZoomIn: document.getElementById('mapZoomIn'),
        mapZoomOut: document.getElementById('mapZoomOut'),
        mapFilterToggle: document.getElementById('mapFilterToggle'),
        mapFilterPanel: document.getElementById('mapFilterPanel'),
        mapFilterClose: document.getElementById('mapFilterClose'),
        mapFullscreenToggle: document.getElementById('mapFullscreenToggle'),
        fullscreenModal: document.getElementById('fullscreenModal'),
        modalBackdrop: document.getElementById('modalBackdrop'),
        modalTitle: document.getElementById('modalTitle'),
        modalSubtitle: document.getElementById('modalSubtitle'),
        modalBody: document.getElementById('modalBody'),
        closeModalBtn: document.getElementById('closeModalBtn'),
    };

    const graphInfoData = {
        'Single Category, Single Station Timeline': {
            desc: 'A standard timeline visualizing a single variable for one station over time. Ideal for spotting long-term trends and seasonality for a specific metric.',
            rules: 'Exactly 1 row. Specify one station and one feature.',
            allowMultiFeature: false,
            allowMultiRow: false
        },
        'Multiple Categories, Single Station Timeline': {
            desc: 'Overlays multiple different variables on the same timeline for a single station. Values are normalized (0 to 1) to allow direct comparison between different units (e.g., comparing Discharge vs. Water Level).',
            rules: 'Multiple rows allowed. All rows must use the same station.',
            allowMultiFeature: true,
            allowMultiRow: false
        },
        'Single Category Across Multiple Stations Comparison': {
            desc: 'Plots the same feature from various stations on a single chart. Perfect for observing spatial differences within the basin during the same time period.',
            rules: 'Select one feature to see available stations, then choose multiple stations.',
            allowMultiFeature: false,
            allowMultiRow: false,
            isFeatureFirst: true
        },
        'Multiple Categories Across Multiple Stations Comparison': {
            desc: 'The most flexible graph. Normalizes data so you can compare any combination of stations and features to find complex basin-wide correlations.',
            rules: 'Multiple rows allowed. Mix stations and features freely.',
            allowMultiFeature: true,
            allowMultiRow: true
        },
        'Year-over-Year Comparison': {
            desc: 'Groups timeline data by calendar month, plotting a separate line for each year. Makes it easy to compare wet/dry season variations across different years.',
            rules: 'Exactly 1 row. Single station and feature required.',
            allowMultiFeature: false,
            allowMultiRow: false
        },
        'Annual Monthly Totals Overview': {
            desc: 'Aggregates the total sum of a feature strictly by expected calendar month, shown as a bar chart. Usually used for cumulative metrics like Precipitation.',
            rules: 'Exactly 1 row. Single station and feature required.',
            allowMultiFeature: false,
            allowMultiRow: false
        },
        'Flow Duration Curve': {
            desc: 'A descending curve showing the percentage of time a value is equalled or exceeded. The standard benchmark chart in hydrology — answers questions like "what discharge is reliable 90% of the time?"',
            rules: 'Exactly 1 row. Single station and feature required.',
            allowMultiFeature: false,
            allowMultiRow: false
        },
        'Monthly Distribution Box Plot': {
            desc: 'Displays 12 box plots (Jan–Dec) showing the statistical distribution of values across all years. Each box shows median, quartiles, and outliers — ideal for understanding seasonal variability.',
            rules: 'Exactly 1 row. Single station and feature required.',
            allowMultiFeature: false,
            allowMultiRow: false
        },
        'Multi-Station Temporal Heatmap': {
            desc: 'A 2D heatmap grid with stations on the Y-axis and time periods on the X-axis. Cell colour intensity represents the monthly average value — reveals spatial and temporal patterns simultaneously.',
            rules: 'Select one feature, then choose multiple stations to compare.',
            allowMultiFeature: false,
            allowMultiRow: false,
            isFeatureFirst: true
        },
        'Correlation Scatter Plot': {
            desc: 'Plots two features against each other as a scatter plot, coloured by season. A regression trend line shows the strength of the relationship (Pearson r). Ideal for exploring how Rainfall drives Discharge.',
            rules: 'Select exactly 2 features from the same station. Use the multi-select to pick both.',
            allowMultiFeature: true,
            allowMultiRow: false
        },
        'Anomaly Detection Chart': {
            desc: 'Shows deviation from the long-term monthly average. Blue bars = above average (wet anomaly), red bars = below average (dry anomaly). A secondary line overlays the raw value for context.',
            rules: 'Exactly 1 row. Single station and feature required.',
            allowMultiFeature: false,
            allowMultiRow: false
        },
    };

    document.addEventListener('DOMContentLoaded', init);

    function applyTheme(theme) {
        const iconEl = document.getElementById('themeIcon');
        const btnEl = document.getElementById('themeToggleBtn');
        if (theme === 'light') {
            document.documentElement.setAttribute('data-theme', 'light');
            if (iconEl) iconEl.textContent = '☽';
            if (btnEl) btnEl.title = 'Switch to dark mode';
        } else {
            document.documentElement.removeAttribute('data-theme');
            if (iconEl) iconEl.textContent = '☀';
            if (btnEl) btnEl.title = 'Switch to light mode';
        }
        localStorage.setItem('hydrovision-theme', theme);
        if (state.tileLayer && state.map) {
            state.map.removeLayer(state.tileLayer);
            state.tileLayer = L.tileLayer(
                theme === 'light'
                    ? 'https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png'
                    : 'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png',
                { attribution: '&copy; OpenStreetMap &copy; CARTO', subdomains: 'abcd', maxZoom: 19 }
            ).addTo(state.map);
        }
    }

    async function init() {
        setEmptyState(els.visualizationCards, 'No visualizations yet. Use the Explore section on the left to add your first chart.');
        setEmptyState(els.analysisCards, 'No analysis cards yet. Build a selection and choose “Add to analyse”.');
        setEmptyState(els.predictionCards, 'No predictions yet. Configure a station, feature, and horizon on the left.');
        applyTheme(localStorage.getItem('hydrovision-theme') || 'dark');
        bindEvents();
        await loadBootstrap();
        initMap();
        await loadGeoJson();
        await buildDatasetPicker();
        buildFeatureFilterBar();
        buildDatasetFilterBar();
        populateStationSearch();
        populateGraphTypes();
        populatePredictionControls();
        addSeriesRow();
        syncSeriesBuilderUI();
        refreshMarkerVisibility();
    }

    function invalidateMapAfterTransition() {
        setTimeout(() => { if (state.map) state.map.invalidateSize(); }, 370);
    }

    function bindEvents() {
        els.sidebarToggle.addEventListener('click', () => {
            if (window.innerWidth <= 1100) {
                els.sidebar.classList.toggle('open');
            } else {
                els.sidebar.classList.add('collapsed');
                invalidateMapAfterTransition();
            }
        });

        const sidebarToggleCollapsed = document.getElementById('sidebarToggleCollapsed');
        if (sidebarToggleCollapsed) {
            sidebarToggleCollapsed.addEventListener('click', () => {
                els.sidebar.classList.remove('collapsed');
                invalidateMapAfterTransition();
            });
        }

        document.querySelectorAll('.rail-nav-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                const tab = btn.dataset.railTab;
                els.sidebar.classList.remove('collapsed');
                invalidateMapAfterTransition();
                // Switch to the clicked tab
                document.querySelectorAll('.workspace-pill').forEach(p => p.classList.remove('active'));
                const pill = document.querySelector(`.workspace-pill[data-tab-target="${tab}"]`);
                if (pill) pill.click();
                // Update rail active state
                document.querySelectorAll('.rail-nav-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
            });
        });

        els.flyToMekong.addEventListener('click', () => {
            if (state.geojsonLayer && state.map) {
                const bounds = state.geojsonLayer.getBounds();
                if (bounds.isValid()) state.map.flyToBounds(bounds.pad(0.15), { duration: 1.2 });
            }
        });
        els.flyToLamaH.addEventListener('click', () => {
            if (state.lamahGeojsonLayer && state.map) {
                const bounds = state.lamahGeojsonLayer.getBounds();
                if (bounds.isValid()) state.map.flyToBounds(bounds.pad(0.1), { duration: 1.2 });
            }
        });

        els.focusStationBtn.addEventListener('click', focusStationFromSearch);
        els.clearStationFocusBtn.addEventListener('click', clearStationFocus);
        els.stationSearch.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                e.preventDefault();
                focusStationFromSearch();
            }
        });
        els.mapFullscreenToggle.addEventListener('click', toggleMapFullscreen);
        els.closeModalBtn.addEventListener('click', closeFullscreen);
        els.modalBackdrop.addEventListener('click', closeFullscreen);
        window.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                if (!els.fullscreenModal.classList.contains('hidden')) {
                    closeFullscreen();
                } else if (document.body.classList.contains('map-fullscreen')) {
                    toggleMapFullscreen();
                }
            }
        });
        els.closeStationCardBtn.addEventListener('click', () => {
            els.stationDetailCard.classList.add('hidden');
            els.mapSearchBar.classList.remove('hidden');
        });
        els.graphTypeSelect.addEventListener('change', () => {
            els.graphInfoPopover.classList.add('hidden');
            showMessage(els.builderMessage, '', '');
            syncSeriesBuilderUI();
        });

        els.addSeriesRowBtn.addEventListener('click', () => {
            const type = els.graphTypeSelect.value;
            const info = type && graphInfoData[type];
            if (info && info.allowMultiRow) {
                // Get all currently selected stations
                const selectedStations = new Set();
                els.seriesBuilder.querySelectorAll('.series-station').forEach(sel => {
                    if (sel.value) selectedStations.add(sel.value);
                });
                const allStations = state.bootstrap.station_names;
                const nextStation = allStations.find(s => !selectedStations.has(s));
                if (!nextStation) {
                    showMessage(els.builderMessage, 'All stations are already selected.', 'error');
                    return;
                }
                addSeriesRow({ station: nextStation });
            } else {
                addSeriesRow();
            }
        });
        
        els.graphInfoBtn.addEventListener('click', () => {
             const type = els.graphTypeSelect.value;
             const info = graphInfoData[type];
             if (info) {
                 els.popoverTitle.textContent = type;
                 els.popoverDesc.textContent = info.desc;
                 els.popoverRules.textContent = info.rules;
                 els.graphInfoPopover.classList.toggle('hidden');
             }
        });
        
        els.closePopoverBtn.addEventListener('click', () => {
            els.graphInfoPopover.classList.add('hidden');
        });
        els.addVisualizationBtn.addEventListener('click', () => submitSeriesRequest('visualize'));
        els.addAnalysisBtn.addEventListener('click', () => submitSeriesRequest('analysis'));
        els.clearVisualizationsBtn.addEventListener('click', () => setEmptyState(els.visualizationCards, 'No visualizations yet. Use the Explore section on the left to add your first chart.'));
        els.clearAnalysisBtn.addEventListener('click', () => setEmptyState(els.analysisCards, 'No analysis cards yet. Build a selection and choose “Add to analyse”.'));
        els.clearPredictionBtn.addEventListener('click', () => setEmptyState(els.predictionCards, 'No predictions yet. Configure a station, feature, and horizon on the left.'));

        els.dockTabs.forEach((tab) => {
            tab.addEventListener('click', () => activateDockTab(tab.dataset.dockTab));
        });
        els.workspacePills.forEach((pill) => {
            pill.addEventListener('click', () => activateDockTab(pill.dataset.tabTarget));
        });

        els.predictStationSelect.addEventListener('change', updatePredictionFeatureOptions);
        els.predictFeatureSelect.addEventListener('change', updatePredictionHint);
        els.runPredictionBtn.addEventListener('click', runPrediction);

        // Theme toggle
        const themeToggleBtn = document.getElementById('themeToggleBtn');
        if (themeToggleBtn) {
            themeToggleBtn.addEventListener('click', () => {
                const isLight = document.documentElement.getAttribute('data-theme') === 'light';
                applyTheme(isLight ? 'dark' : 'light');
            });
        }

        initDockResizer();

        els.mapZoomIn.addEventListener('click', () => state.map.zoomIn());
        els.mapZoomOut.addEventListener('click', () => state.map.zoomOut());
        els.mapFilterToggle.addEventListener('click', () => {
            const isHidden = els.mapFilterPanel.classList.toggle('hidden');
            els.mapFilterToggle.classList.toggle('active', !isHidden);
        });
        els.mapFilterClose.addEventListener('click', () => {
            els.mapFilterPanel.classList.add('hidden');
            els.mapFilterToggle.classList.remove('active');
        });
    }

    async function loadBootstrap() {
        const response = await fetch('/api/bootstrap');
        state.bootstrap = await response.json();
        state.bootstrap.stations.forEach((station) => state.stationsByName.set(station.station, station));
    }

    function initMap() {
        state.map = L.map('map', {
            zoomControl: false,
            worldCopyJump: true,
            attributionControl: false,
        }).setView([20, 103], 5);

        const isDark = document.documentElement.getAttribute('data-theme') !== 'light';
        state.tileLayer = L.tileLayer(
            isDark
                ? 'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png'
                : 'https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png',
            { attribution: '&copy; OpenStreetMap &copy; CARTO', subdomains: 'abcd', maxZoom: 19 }
        ).addTo(state.map);

        state.bootstrap.stations.forEach((station) => {
            const marker = L.circleMarker([station.lat, station.lon], { ...markerStyleForStation(station.station), interactive: true });
            marker.bindTooltip(buildTooltip(station), {
                direction: 'top',
                className: 'station-tooltip',
                sticky: false,
                opacity: 1,
                offset: [0, -8],
            });
            marker.on('mouseover', () => marker.setStyle({ radius: markerStyleForStation(station.station).radius + 1 }));
            marker.on('mouseout', () => marker.setStyle(markerStyleForStation(station.station)));
            marker.on('click', () => focusStation(station.station, true));
            marker.addTo(state.map);
            state.markers.set(station.station, marker);
        });
    }

    async function loadGeoJson() {
        const [mekongRes, lamahRes] = await Promise.all([
            fetch('/api/mekong-geojson'),
            fetch('/api/lamah-geojson'),
        ]);
        const mekongGeojson = await mekongRes.json();
        const lamahGeojson = await lamahRes.json();

        state.geojsonLayer = L.geoJSON(mekongGeojson, {
            interactive: false,
            style: {
                color: '#38bdf8',
                weight: 1.5,
                opacity: 0.7,
                fillColor: '#38bdf8',
                fillOpacity: 0.08,
            },
        }).addTo(state.map);

        // Default view: focus on Mekong basin on load
        const mekongBounds = state.geojsonLayer.getBounds();
        if (mekongBounds.isValid()) state.map.fitBounds(mekongBounds.pad(0.15));

        state.lamahGeojsonLayer = L.geoJSON(lamahGeojson, {
            interactive: false,
            style: {
                weight: 0,
                fillColor: '#f2bd41',
                fillOpacity: 0.2,
            },
        }).addTo(state.map);
    }

    async function buildDatasetPicker() {
        if (!els.datasetPicker) return;
        const res = await fetch('/api/datasets');
        const folderNames = await res.json(); // e.g. ["LamaH", "Mekong"]

        // Map folder name → dataset key used in bootstrap data
        const keyMap = { lamah: 'lamah', mekong: 'mekong' };
        const datasets = folderNames.map(name => ({
            label: name,
            key: keyMap[name.toLowerCase()] ?? name.toLowerCase(),
        }));

        state.activeDatasetFilter = 'all';

        els.datasetPicker.innerHTML = '';
        const allDatasets = [{ label: 'All', key: 'all' }, ...datasets];
        allDatasets.forEach(({ label, key }) => {
            const btn = document.createElement('button');
            btn.className = `feature-chip${key === state.activeDatasetFilter ? ' active' : ''}`;
            btn.textContent = label;
            btn.addEventListener('click', () => {
                if (state.activeDatasetFilter === key) return;
                state.activeDatasetFilter = key;
                Array.from(els.datasetPicker.children).forEach(c => c.classList.remove('active'));
                btn.classList.add('active');
                // Sync the map filter bar too
                Array.from(els.datasetFilterBar?.children ?? []).forEach(c => {
                    c.classList.toggle('active', c.textContent.toLowerCase().includes(key) || (key === 'all' && c.textContent.toLowerCase() === 'all datasets'));
                });
                buildFeatureFilterBar();
                refreshMarkerVisibility();
                // Refresh station dropdowns in the series builder
                els.seriesBuilder.querySelectorAll('.series-station').forEach(sel => {
                    fillStationSelect(sel, sel.value);
                });
                // Fly map to the chosen dataset
                if (key === 'mekong' && state.geojsonLayer) {
                    const b = state.geojsonLayer.getBounds();
                    if (b.isValid()) state.map.flyToBounds(b.pad(0.15), { duration: 1.2 });
                } else if (key === 'lamah' && state.lamahGeojsonLayer) {
                    const b = state.lamahGeojsonLayer.getBounds();
                    if (b.isValid()) state.map.flyToBounds(b.pad(0.1), { duration: 1.2 });
                }
            });
            els.datasetPicker.appendChild(btn);
        });
    }

    function buildFeatureFilterBar() {
        els.featureFilterBar.innerHTML = '';

        // Show only features relevant to the active dataset
        const ds = state.activeDatasetFilter;
        const datasetFeatures = state.bootstrap.dataset_features || {};
        const datasetCounts = state.bootstrap.dataset_feature_counts || {};

        let relevantFeatures;
        let countMap;
        if (ds === 'all') {
            relevantFeatures = state.bootstrap.features;
            countMap = state.bootstrap.feature_counts;
        } else {
            relevantFeatures = datasetFeatures[ds] || [];
            countMap = datasetCounts[ds] || {};
        }

        // If current filter no longer applies, reset to All
        if (state.activeFeatureFilter !== 'All' && !relevantFeatures.includes(state.activeFeatureFilter)) {
            state.activeFeatureFilter = 'All';
        }

        ['All', ...relevantFeatures].forEach((feature) => {
            const btn = document.createElement('button');
            btn.className = `feature-chip${feature === state.activeFeatureFilter ? ' active' : ''}`;
            btn.textContent = feature === 'All'
                ? 'All stations'
                : `${prettyFeature(feature)} (${countMap[feature] || 0})`;
            btn.addEventListener('click', () => {
                state.activeFeatureFilter = feature;
                Array.from(els.featureFilterBar.children).forEach((child) => child.classList.remove('active'));
                btn.classList.add('active');
                refreshMarkerVisibility();
            });
            els.featureFilterBar.appendChild(btn);
        });
    }

    function buildDatasetFilterBar() {
        if (!els.datasetFilterBar) return;
        els.datasetFilterBar.innerHTML = '';
        const datasets = [
            { key: 'all', label: 'All datasets' },
            { key: 'mekong', label: 'Mekong' },
            { key: 'lamah', label: 'LamaH-CE' },
        ];
        datasets.forEach(({ key, label }) => {
            const btn = document.createElement('button');
            btn.className = `feature-chip${key === state.activeDatasetFilter ? ' active' : ''}`;
            btn.textContent = label;
            btn.addEventListener('click', () => {
                state.activeDatasetFilter = key;
                Array.from(els.datasetFilterBar.children).forEach((c) => c.classList.remove('active'));
                btn.classList.add('active');
                buildFeatureFilterBar();
                refreshMarkerVisibility();
            });
            els.datasetFilterBar.appendChild(btn);
        });
    }

    function populateStationSearch() {
        const input = els.stationSearch;
        const list = els.stationSuggestList;

        const mekongNames = state.bootstrap.station_names.filter(n => {
            const s = state.stationsByName.get(n);
            return s && s.dataset === 'mekong';
        });

        function showSuggestions(query) {
            const q = query.trim().toLowerCase();
            const pool = q.length === 0 ? mekongNames : state.bootstrap.station_names;
            const matches = q.length === 0
                ? pool
                : pool.filter(n => prettyStation(n).toLowerCase().includes(q) || n.toLowerCase().includes(q)).slice(0, 80);

            list.innerHTML = '';
            if (matches.length === 0) { list.classList.add('hidden'); return; }
            matches.forEach(name => {
                const item = document.createElement('div');
                item.className = 'station-suggest-item';
                item.textContent = prettyStation(name);
                item.addEventListener('mousedown', (e) => {
                    e.preventDefault();
                    input.value = prettyStation(name);
                    list.classList.add('hidden');
                    focusStation(name, true);
                });
                list.appendChild(item);
            });
            if (q.length === 0) {
                const hint = document.createElement('div');
                hint.className = 'station-suggest-hint';
                hint.textContent = `Type a number to search LamaH-CE stations…`;
                list.appendChild(hint);
            }
            list.classList.remove('hidden');
        }

        input.addEventListener('input', () => showSuggestions(input.value));
        input.addEventListener('focus', () => showSuggestions(input.value));
        input.addEventListener('blur', () => setTimeout(() => list.classList.add('hidden'), 150));
    }

    function populateGraphTypes() {
        els.graphTypeSelect.innerHTML = '';
        state.bootstrap.graph_types.forEach((graphType) => {
            const option = document.createElement('option');
            option.value = graphType;
            option.textContent = graphType;
            els.graphTypeSelect.appendChild(option);
        });
    }

    function populatePredictionControls() {
        fillStationSelect(els.predictStationSelect);
        updatePredictionFeatureOptions();
    }

    function updatePredictionFeatureOptions() {
        const station = els.predictStationSelect.value;
        const meta = state.stationsByName.get(station);
        els.predictFeatureSelect.innerHTML = '';
        (meta?.features || []).forEach((feature) => {
            const option = document.createElement('option');
            option.value = feature;
            option.textContent = prettyFeature(feature);
            els.predictFeatureSelect.appendChild(option);
        });
        updatePredictionHint();
    }

    function updatePredictionHint() {
        const station = els.predictStationSelect.value;
        const feature = els.predictFeatureSelect.value;
        const meta = state.stationsByName.get(station);
        if (!meta || !feature) {
            els.predictHint.textContent = 'Choose a station and feature to see prediction details.';
            return;
        }
        const detail = meta.feature_details[feature];
        els.predictHint.textContent = `Available ${prettyFeature(feature)} data spans ${detail.start_date} → ${detail.end_date}. Frequency: ${detail.frequency}. Latest value: ${formatNumber(detail.latest_value)} ${detail.unit}.`;
    }

    function syncSeriesBuilderUI() {
        if (!els.graphTypeSelect.value) return;
        const info = graphInfoData[els.graphTypeSelect.value];
        if (!info) return;

        // ------- Feature-first mode (Single Category Across Multiple Stations) -------
        if (info.isFeatureFirst) {
            els.addSeriesRowBtn.classList.add('hidden');
            // If not already in feature-first mode, swap the builder contents
            if (!els.seriesBuilder.querySelector('.feature-first-container')) {
                els.seriesBuilder.innerHTML = '';
                buildFeatureFirstUI();
            }
            return;
        }

        // ------- Normal mode -------
        // If we were previously in feature-first mode, tear it down and add a normal row
        if (els.seriesBuilder.querySelector('.feature-first-container')) {
            els.seriesBuilder.innerHTML = '';
            addSeriesRow();
        }

        if (info.allowMultiRow) {
            els.addSeriesRowBtn.classList.remove('hidden');
        } else {
            els.addSeriesRowBtn.classList.add('hidden');
            const rows = Array.from(els.seriesBuilder.querySelectorAll('.series-row'));
            for (let i = 1; i < rows.length; i++) {
                rows[i].remove();
            }
        }
        
        updateRemoveButtonsVisibility();

        const isAnnualOverview = els.graphTypeSelect.value === 'Annual Monthly Totals Overview';
        const usePerFeatureDates = info.allowMultiFeature && (info.allowMultiRow || isAnnualOverview);

        const featureSelects = els.seriesBuilder.querySelectorAll('.series-feature');
        featureSelects.forEach(select => {
            const group = select.closest('.control-group');
            const row = select.closest('.series-row');
            if (info.allowMultiFeature) {
                select.setAttribute('multiple', 'multiple');
                select.setAttribute('size', select.options.length || 2);
                select.style.height = 'auto';
                if (group) group.classList.add('full-span');
            } else {
                select.removeAttribute('multiple');
                select.removeAttribute('size'); // CRITICAL: remove size attribute so it becomes a normal dropdown again
                select.style.height = 'auto';
                if (group) group.classList.remove('full-span');
                if (select.selectedOptions.length > 1) {
                    const firstVal = select.selectedOptions[0].value;
                    select.value = firstVal;
                }
            }

            // Toggle shared vs per-feature dates
            if (row) {
                row.querySelectorAll('.shared-dates').forEach(el => {
                    if (usePerFeatureDates) el.classList.add('hidden');
                    else el.classList.remove('hidden');
                });
                const pfContainer = row.querySelector('.per-feature-dates');
                if (pfContainer) {
                    if (usePerFeatureDates) {
                        pfContainer.classList.remove('hidden');
                        buildPerFeatureDates(row);
                    } else {
                        pfContainer.classList.add('hidden');
                        pfContainer.innerHTML = '';
                    } 
                }
                
                // Re-sync date bounds to force type changes (e.g., date -> month)
                syncRowDateBounds(row);
            }
        });
    }

    function updateRemoveButtonsVisibility() {
        const rows = els.seriesBuilder.querySelectorAll('.series-row');
        const showDelete = rows.length > 1;
        rows.forEach(row => {
            const btn = row.querySelector('.series-remove-btn');
            if (btn) {
                if (showDelete) btn.classList.remove('hidden');
                else btn.classList.add('hidden');
            }
        });
    }

    function addSeriesRow(prefill) {
        const fragment = els.seriesRowTemplate.content.cloneNode(true);
        const row = fragment.querySelector('.series-row');
        const stationSelect = fragment.querySelector('.series-station');
        const featureSelect = fragment.querySelector('.series-feature');
        const startInput = fragment.querySelector('.series-start');
        const endInput = fragment.querySelector('.series-end');
        const metaLabel = fragment.querySelector('.series-row-meta');
        const removeBtn = row.querySelector('.series-remove-btn');

        fillStationSelect(stationSelect, prefill?.station);
        let previousStationValue = stationSelect.value;
        
        stationSelect.addEventListener('change', () => {
            const info = graphInfoData[els.graphTypeSelect.value];
            if (info && info.allowMultiRow && !info.isFeatureFirst && !stationSelect.multiple) {
                // Prevent duplicate station selection across rows
                const currentVal = stationSelect.value;
                const rows = Array.from(els.seriesBuilder.querySelectorAll('.series-row'));
                const isDuplicate = rows.some(r => r !== row && r.querySelector('.series-station').value === currentVal);
                
                if (isDuplicate) {
                    showMessage(els.builderMessage, 'This station is already selected in another row.', 'error');
                    stationSelect.value = previousStationValue; // Revert
                    return;
                }
            }
            previousStationValue = stationSelect.value;

            if (els.graphTypeSelect.value && graphInfoData[els.graphTypeSelect.value].isFeatureFirst) {
                syncRowDateBounds(row, true);
            } else {
                syncRowFeatureOptions(row);
                const isAnnualOverview = els.graphTypeSelect.value === 'Annual Monthly Totals Overview';
                if (info && info.allowMultiFeature && (info.allowMultiRow || isAnnualOverview)) {
                    buildPerFeatureDates(row);
                } else {
                    syncRowDateBounds(row, true);
                }
            }
        });
        
        stationSelect.addEventListener('mousedown', function(e) {
            if (!this.multiple) return;
            if (e.target.tagName === 'OPTION') {
                e.preventDefault();
                e.target.selected = !e.target.selected;
                const event = new Event('change', { bubbles: true });
                this.dispatchEvent(event);
                setTimeout(() => this.focus(), 0);
            }
        });

        featureSelect.addEventListener('change', () => {
            const info = graphInfoData[els.graphTypeSelect.value];
            const isAnnualOverview = els.graphTypeSelect.value === 'Annual Monthly Totals Overview';
            if (info && info.isFeatureFirst) {
                syncRowStationOptions(row);
                syncRowDateBounds(row, true);
            } else if (info && info.allowMultiFeature && (info.allowMultiRow || isAnnualOverview)) {
                // Rebuild per-feature date rows
                buildPerFeatureDates(row);
            } else {
                syncRowDateBounds(row, true);
            }
        });
        featureSelect.addEventListener('mousedown', function(e) {
            if (!this.multiple) return;
            if (e.target.tagName === 'OPTION') {
                e.preventDefault();
                e.target.selected = !e.target.selected;
                const event = new Event('change', { bubbles: true });
                this.dispatchEvent(event);
                setTimeout(() => this.focus(), 0);
            }
        });

        if (removeBtn) {
            removeBtn.addEventListener('click', () => {
                row.remove();
                updateRemoveButtonsVisibility();
            });
        }

        els.seriesBuilder.appendChild(fragment);
        const insertedRow = els.seriesBuilder.lastElementChild;
        if (prefill?.feature) {
            syncRowFeatureOptions(insertedRow, prefill.feature);
        } else {
            syncRowFeatureOptions(insertedRow);
        }
        if (prefill?.start_date) insertedRow.querySelector('.series-start').value = prefill.start_date;
        if (prefill?.end_date) insertedRow.querySelector('.series-end').value = prefill.end_date;
        syncRowDateBounds(insertedRow);
        metaLabel.textContent = 'The row metadata updates automatically based on the station and feature you choose.';
        
        setTimeout(syncSeriesBuilderUI, 0); 
    }

    function fillStationSelect(selectEl, selectedValue) {
        selectEl.innerHTML = '';
        const ds = state.activeDatasetFilter;
        state.bootstrap.station_names.forEach((stationName) => {
            if (ds !== 'all') {
                const meta = state.stationsByName.get(stationName);
                if (!meta || meta.dataset !== ds) return;
            }
            const option = document.createElement('option');
            option.value = stationName;
            option.textContent = prettyStation(stationName);
            if (selectedValue && selectedValue === stationName) {
                option.selected = true;
            }
            selectEl.appendChild(option);
        });
    }

    function syncRowFeatureOptions(row, selectedFeature) {
        const station = row.querySelector('.series-station').value;
        const featureSelect = row.querySelector('.series-feature');
        const meta = state.stationsByName.get(station);
        const features = meta?.features || [];
        featureSelect.innerHTML = '';
        features.forEach((feature) => {
            const option = document.createElement('option');
            option.value = feature;
            option.textContent = prettyFeature(feature);
            if (selectedFeature && selectedFeature === feature) {
                option.selected = true;
            }
            featureSelect.appendChild(option);
        });
        if (featureSelect.multiple) {
            featureSelect.setAttribute('size', features.length || 2);
            featureSelect.style.height = 'auto';
            featureSelect.style.overflow = 'scroll';
            requestAnimationFrame(() => {
                featureSelect.style.height = featureSelect.scrollHeight + 'px';
                featureSelect.style.overflow = 'hidden';
            });
        }
        syncRowDateBounds(row);
    }

    function syncRowStationOptions(row) {
        const feature = row.querySelector('.series-feature').value;
        const stationSelect = row.querySelector('.series-station');
        const prevSelections = Array.from(stationSelect.selectedOptions).map(opt => opt.value);
        
        stationSelect.innerHTML = '';
        if (!feature) return;

        state.bootstrap.station_names.forEach((stationName) => {
            const meta = state.stationsByName.get(stationName);
            if (meta && meta.features.includes(feature)) {
                const option = document.createElement('option');
                option.value = stationName;
                option.textContent = prettyStation(stationName);
                if (prevSelections.includes(stationName)) {
                    option.selected = true;
                }
                stationSelect.appendChild(option);
            }
        });
        // Important: we don't sync row date bounds immediately here because we just wiped stations. 
        // We let the user pick stations, and the change event will trigger syncRowDateBounds.
        // Wait, start_date and end_date are bound to the combination of 1 station + 1 feature.
        // For multiple stations, we'll bound it by the *first* selected station, or full available range for the feature at ALL stations.
        // Or we just call syncRowDateBounds and let it handle the multi-station case.
        syncRowDateBounds(row);
    }

    function getFullMonthBounds(startStr, endStr) {
        let [startY, startM, startD] = startStr.split('-').map(Number);
        let [endY, endM, endD] = endStr.split('-').map(Number);

        if (startD > 1) {
            startM++;
            if (startM > 12) { startM = 1; startY++; }
        }

        const daysInEndMonth = new Date(endY, endM, 0).getDate();
        if (endD < daysInEndMonth) {
            endM--;
            if (endM < 1) { endM = 12; endY--; }
        }

        const pad = n => n.toString().padStart(2, '0');
        return {
            minMonth: `${startY}-${pad(startM)}`,
            maxMonth: `${endY}-${pad(endM)}`
        };
    }

    function formatMonthLabel(yyyyMmStr) {
        const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
        const [y, m] = yyyyMmStr.split('-');
        return `${months[parseInt(m, 10) - 1]} ${y}`;
    }

    function generateMonthOptions(minMonth, maxMonth) {
        let [minY, minM] = minMonth.split('-').map(Number);
        let [maxY, maxM] = maxMonth.split('-').map(Number);
        const options = [];
        
        while (minY < maxY || (minY === maxY && minM <= maxM)) {
            const val = `${minY}-${minM.toString().padStart(2, '0')}`;
            options.push({ value: val, label: formatMonthLabel(val) });
            minM++;
            if (minM > 12) { minM = 1; minY++; }
        }
        return options;
    }

    function createMonthSelect(el, id, options, selectedVal) {
        if (el.tagName !== 'SELECT') {
            const select = document.createElement('select');
            select.className = el.className;
            select.id = id || '';
            el.parentNode.replaceChild(select, el);
            el = select;
        }
        
        el.innerHTML = '';
        if (options.length === 0) {
            el.disabled = true;
            return el;
        }
        
        options.forEach(opt => {
            const optionEl = document.createElement('option');
            optionEl.value = opt.value;
            optionEl.textContent = opt.label;
            if (opt.value === selectedVal) optionEl.selected = true;
            el.appendChild(optionEl);
        });
        
        el.disabled = false;
        if (!selectedVal || !options.some(o => o.value === selectedVal)) {
            el.value = options[0].value;
        }
        
        return el;
    }

    function forceInputType(inputEl, newType) {
        if (inputEl.tagName === 'SELECT' && newType === 'date') {
            const input = document.createElement('input');
            input.type = 'date';
            input.className = inputEl.className;
            input.id = inputEl.id || '';
            inputEl.parentNode.replaceChild(input, inputEl);
            inputEl = input;
        } else if (inputEl.tagName === 'INPUT' && inputEl.type === newType) {
            return inputEl;
        }

        if (newType !== 'date') return inputEl; 

        // For date inputs, we safely clone to reset events/bindings
        const clone = inputEl.cloneNode(false);
        clone.type = newType;
        clone.value = ''; 
        inputEl.parentNode.replaceChild(clone, inputEl);
        
        clone.addEventListener('change', () => {
            const row = clone.closest('.series-row');
            if (row) {
                const feature = row.querySelector('.series-feature').value;
                const station = row.querySelector('.series-station').value;
                const stationMeta = state.stationsByName.get(station);
                if (stationMeta && stationMeta.feature_details[feature]) {
                    const detail = stationMeta.feature_details[feature];
                    const startInput = row.querySelector('.series-start');
                    const endInput = row.querySelector('.series-end');
                    const isAnnualOverview = els.graphTypeSelect.value === 'Annual Monthly Totals Overview';
                    
                    if (!isAnnualOverview) {
                        if (!startInput.value || startInput.value < detail.start_date || startInput.value > detail.end_date) startInput.value = detail.start_date;
                        if (!endInput.value || endInput.value > detail.end_date || endInput.value < detail.start_date) endInput.value = detail.end_date;
                        if (startInput.value > endInput.value) {
                            startInput.value = detail.start_date;
                            endInput.value = detail.end_date;
                        }
                    }
                }
            }
        });
        
        return clone;
    }

    function syncRowDateBounds(row, forceReset = false) {
        let station = row.querySelector('.series-station').value;
        const feature = row.querySelector('.series-feature').value;
        let startInput = row.querySelector('.series-start');
        let endInput = row.querySelector('.series-end');
        const metaEl = row.querySelector('.series-row-meta');
        
        const stationSelect = row.querySelector('.series-station');
        if (stationSelect.multiple && stationSelect.selectedOptions.length > 0) {
            station = stationSelect.selectedOptions[0].value;
        }

        const stationMeta = state.stationsByName.get(station);
        if (!stationMeta || !feature || !stationMeta.feature_details[feature]) {
            metaEl.textContent = 'Please select inputs.';
            return;
        }
        
        const detail = stationMeta.feature_details[feature];
        const isAnnualOverview = els.graphTypeSelect.value === 'Annual Monthly Totals Overview';

        if (isAnnualOverview) {
            const { minMonth, maxMonth } = getFullMonthBounds(detail.start_date, detail.end_date);
            
            if (minMonth > maxMonth) {
                startInput = createMonthSelect(startInput, startInput.id, []);
                endInput = createMonthSelect(endInput, endInput.id, []);
                metaEl.textContent = 'No full calendar months available for this feature.';
                return;
            }

            const options = generateMonthOptions(minMonth, maxMonth);
            
            // On first switch, the input values will be YYYY-MM-DD, which won't match our YYYY-MM options.
            // We strip them to YYYY-MM, or default to the min/max month if empty.
            let prevStart = startInput.value ? startInput.value.slice(0, 7) : minMonth;
            let prevEnd = endInput.value ? endInput.value.slice(0, 7) : maxMonth;
            
            if (forceReset) {
                prevStart = minMonth;
                prevEnd = maxMonth;
            } else {
                // If the old YYYY-MM is outside the valid bounds, fallback to the boundaries.
                if (prevStart < minMonth || prevStart > maxMonth) prevStart = minMonth;
                if (prevEnd > maxMonth || prevEnd < minMonth) prevEnd = maxMonth;
            }
            
            startInput = createMonthSelect(startInput, startInput.id, options, prevStart);
            endInput = createMonthSelect(endInput, endInput.id, options, prevEnd);

            // Re-bind change listeners since we replaced elements
            const enforceMonthBounds = () => {
                if (startInput.value > endInput.value) {
                    startInput.value = minMonth;
                    endInput.value = maxMonth;
                }
            };
            startInput.addEventListener('change', enforceMonthBounds);
            endInput.addEventListener('change', enforceMonthBounds);

            // Force initial bound enforcement
            enforceMonthBounds();

            metaEl.textContent = `Full months available: ${formatMonthLabel(minMonth)} → ${formatMonthLabel(maxMonth)} · ${detail.unit}`;
        } else {
            startInput = forceInputType(startInput, 'date');
            endInput = forceInputType(endInput, 'date');
            startInput.disabled = false;
            endInput.disabled = false;
            startInput.min = detail.start_date;
            startInput.max = detail.end_date;
            endInput.min = detail.start_date;
            endInput.max = detail.end_date;
            if (!startInput.value || startInput.value < detail.start_date || startInput.value > detail.end_date) {
                startInput.value = detail.start_date;
            }
            if (!endInput.value || endInput.value > detail.end_date || endInput.value < detail.start_date) {
                endInput.value = detail.end_date;
            }
            if (startInput.value > endInput.value) {
                startInput.value = detail.start_date;
                endInput.value = detail.end_date;
            }

            if (stationSelect.multiple && stationSelect.selectedOptions.length > 1) {
                metaEl.textContent = `Coverage approx based on first station: ${detail.start_date} → ${detail.end_date}`;
            } else {
                metaEl.textContent = `Coverage: ${detail.start_date} → ${detail.end_date} · ${detail.observations.toLocaleString()} observations · ${detail.imputed_points.toLocaleString()} imputed · Unit: ${detail.unit}`;
            }
        }
        // Fallback for regular dates if somehow out of bounds
        if (!isAnnualOverview) {
            if (forceReset) {
                startInput.value = detail.start_date;
                endInput.value = detail.end_date;
            } else {
                if (!startInput.value || startInput.value < detail.start_date || startInput.value > detail.end_date) {
                    startInput.value = detail.start_date;
                }
                if (!endInput.value || endInput.value > detail.end_date || endInput.value < detail.start_date) {
                    endInput.value = detail.end_date;
                }
                if (startInput.value > endInput.value) {
                    startInput.value = detail.start_date;
                    endInput.value = detail.end_date;
                }
            }
        }

        if (stationSelect.multiple && stationSelect.selectedOptions.length > 1) {
            metaEl.textContent = `Coverage approx based on first station: ${detail.start_date} → ${detail.end_date}`;
        } else {
            metaEl.textContent = `Coverage: ${detail.start_date} → ${detail.end_date} · ${detail.observations.toLocaleString()} observations · ${detail.imputed_points.toLocaleString()} imputed · Unit: ${detail.unit}`;
        }
    }

    function buildPerFeatureDates(row) {
        const pfContainer = row.querySelector('.per-feature-dates');
        if (!pfContainer) return;
        
        const station = row.querySelector('.series-station').value;
        const featureSelect = row.querySelector('.series-feature');
        const selectedFeatures = Array.from(featureSelect.selectedOptions).map(opt => opt.value);
        
        pfContainer.innerHTML = ''; // Clear existing
        const stationMeta = state.stationsByName.get(station);
        
        if (!station || !stationMeta || selectedFeatures.length === 0) {
            return;
        }

        selectedFeatures.forEach(feature => {
            const detail = stationMeta.feature_details[feature];
            if (!detail) return;

            const div = document.createElement('div');
            div.className = 'pf-date-row';
            div.dataset.feature = feature;
            div.style.marginTop = '10px';
            div.style.borderTop = '1px solid var(--line)';
            div.style.padding = '10px 14px 0 14px';

            div.innerHTML = `
                <div class="control-group" style="margin-bottom: 8px;">
                    <label style="color: var(--primary);">Dates for ${prettyFeature(feature)}</label>
                    <div style="font-size: 11px; color: var(--text-muted); margin-top: 2px;">Available: ${detail.start_date} → ${detail.end_date}</div>
                </div>
                <div class="control-grid two-col">
                    <div class="control-group">
                        <label>Start date</label>
                        <input class="pf-start" type="date" value="${detail.start_date}" min="${detail.start_date}" max="${detail.end_date}" />
                    </div>
                    <div class="control-group">
                        <label>End date</label>
                        <input class="pf-end" type="date" value="${detail.end_date}" min="${detail.start_date}" max="${detail.end_date}" />
                    </div>
                </div>
            `;

            // Validate dates
            let startInput = div.querySelector('.pf-start');
            let endInput = div.querySelector('.pf-end');
            
            const isAnnualOverview = els.graphTypeSelect.value === 'Annual Monthly Totals Overview';
            if (isAnnualOverview) {
                const { minMonth, maxMonth } = getFullMonthBounds(detail.start_date, detail.end_date);
                if (minMonth > maxMonth) {
                    startInput = createMonthSelect(startInput, startInput.id, []);
                    endInput = createMonthSelect(endInput, endInput.id, []);
                } else {
                    const options = generateMonthOptions(minMonth, maxMonth);
                    
                    // Always initialize a freshly built row to its min and max bounds
                    let prevStart = minMonth;
                    let prevEnd = maxMonth;
                    
                    startInput = createMonthSelect(startInput, startInput.id, options, prevStart);
                    endInput = createMonthSelect(endInput, endInput.id, options, prevEnd);
                    
                    const enforceMonthBounds = () => {
                        if (startInput.value > endInput.value) {
                            startInput.value = minMonth;
                            endInput.value = maxMonth;
                        }
                    };
                    startInput.addEventListener('change', enforceMonthBounds);
                    endInput.addEventListener('change', enforceMonthBounds);
                }
            } else {
                startInput = forceInputType(startInput, 'date');
                endInput = forceInputType(endInput, 'date');
                
                const enforceBounds = () => {
                    if (!startInput.value || startInput.value < detail.start_date || startInput.value > detail.end_date) startInput.value = detail.start_date;
                    if (!endInput.value || endInput.value > detail.end_date || endInput.value < detail.start_date) endInput.value = detail.end_date;
                    if (startInput.value > endInput.value) {
                        startInput.value = detail.start_date;
                        endInput.value = detail.end_date;
                    }
                };

                startInput.addEventListener('change', enforceBounds);
                endInput.addEventListener('change', enforceBounds);
            }

            pfContainer.appendChild(div);
        });
    }

    function formatOutputDate(dateStr, isEnd) {
        if (!dateStr) return '';
        if (dateStr.length === 7) {
            // It's a YYYY-MM string
            if (isEnd) {
                const [y, m] = dateStr.split('-').map(Number);
                const lastDay = new Date(y, m, 0).getDate();
                return `${dateStr}-${lastDay.toString().padStart(2, '0')}`;
            } else {
                return `${dateStr}-01`;
            }
        }
        return dateStr;
    }

    function getSeriesSelections() {
        // ------- Feature-first mode -------
        const ffContainer = els.seriesBuilder.querySelector('.feature-first-container');
        if (ffContainer) {
            const feature = ffContainer.querySelector('.ff-feature-select').value;
            const selections = [];
            ffContainer.querySelectorAll('.station-row-item').forEach(row => {
                const station = row.querySelector('.sr-station-select').value;
                const start_date = formatOutputDate(row.querySelector('.sr-start').value, false);
                const end_date = formatOutputDate(row.querySelector('.sr-end').value, true);
                if (station && start_date && end_date) {
                    selections.push({ station, feature, start_date, end_date });
                }
            });
            return selections;
        }

        // ------- Normal mode -------
        const selections = [];
        Array.from(els.seriesBuilder.querySelectorAll('.series-row')).forEach((row) => {
            const stationSelect = row.querySelector('.series-station');
            const featureSelect = row.querySelector('.series-feature');
            const selectedStations = Array.from(stationSelect.selectedOptions).map(opt => opt.value);
            const selectedFeatures = Array.from(featureSelect.selectedOptions).map(opt => opt.value);

            // Check if per-feature dates are active
            const pfContainer = row.querySelector('.per-feature-dates');
            if (pfContainer && !pfContainer.classList.contains('hidden') && pfContainer.children.length > 0) {
                // Per-feature date mode
                selectedStations.forEach(station => {
                    pfContainer.querySelectorAll('.pf-date-row').forEach(pfRow => {
                        const feature = pfRow.dataset.feature;
                        const start_date = formatOutputDate(pfRow.querySelector('.pf-start').value, false);
                        const end_date = formatOutputDate(pfRow.querySelector('.pf-end').value, true);
                        if (feature && start_date && end_date) {
                            selections.push({ station, feature, start_date, end_date });
                        }
                    });
                });
            } else {
                // Shared date mode
                const start_date = formatOutputDate(row.querySelector('.series-start').value, false);
                const end_date = formatOutputDate(row.querySelector('.series-end').value, true);
                selectedStations.forEach(station => {
                    selectedFeatures.forEach(feature => {
                        selections.push({ station, feature, start_date, end_date });
                    });
                });
            }
        });
        return selections;
    }

    // ==================== Feature-First Mode ====================

    function buildFeatureFirstUI() {
        const fragment = els.featureFirstTemplate.content.cloneNode(true);
        const container = fragment.querySelector('.feature-first-container');
        const featureSelect = fragment.querySelector('.ff-feature-select');
        const addBtn = fragment.querySelector('.ff-add-station-btn');

        // Populate feature dropdown with all features
        state.bootstrap.features.forEach(feature => {
            const option = document.createElement('option');
            option.value = feature;
            option.textContent = prettyFeature(feature);
            featureSelect.appendChild(option);
        });

        featureSelect.addEventListener('change', () => {
            // When feature changes, reset to a single station row
            const stationRowsContainer = container.querySelector('.ff-station-rows');
            stationRowsContainer.innerHTML = '';
            ffAddStationRow(container, featureSelect.value);
        });

        addBtn.addEventListener('click', () => {
            const feature = featureSelect.value;
            // Get all currently selected stations
            const selectedStations = new Set();
            container.querySelectorAll('.sr-station-select').forEach(sel => {
                if (sel.value) selectedStations.add(sel.value);
            });
            // Get all available stations for this feature
            const availableStations = state.bootstrap.station_names.filter(name => {
                const meta = state.stationsByName.get(name);
                return meta && meta.features.includes(feature);
            });
            // Find first unselected station
            const nextStation = availableStations.find(s => !selectedStations.has(s));
            if (!nextStation) {
                // All stations are already selected - show red notification
                ffShowNotification(container, 'All stations for this feature are already selected.');
                return;
            }
            ffAddStationRow(container, feature, nextStation);
        });

        els.seriesBuilder.appendChild(fragment);

        // Add one default station row
        const insertedContainer = els.seriesBuilder.querySelector('.feature-first-container');
        ffAddStationRow(insertedContainer, insertedContainer.querySelector('.ff-feature-select').value);
    }

    function ffShowNotification(container, message) {
        // Remove any existing notification
        const existing = container.querySelector('.ff-notification');
        if (existing) existing.remove();
        const div = document.createElement('div');
        div.className = 'ff-notification';
        div.textContent = message;
        div.style.cssText = 'color:#fca5a5;font-size:13px;padding:8px 0;font-weight:500;';
        const addBtn = container.querySelector('.ff-add-station-btn');
        addBtn.parentNode.insertBefore(div, addBtn.nextSibling);
        setTimeout(() => div.remove(), 3000);
    }

    function ffAddStationRow(container, feature, preselect) {
        const fragment = els.stationRowTemplate.content.cloneNode(true);
        const row = fragment.querySelector('.station-row-item');
        const stationSelect = row.querySelector('.sr-station-select');
        const removeBtn = row.querySelector('.station-row-remove-btn');
        const stationRowsContainer = container.querySelector('.ff-station-rows');

        // Populate station dropdown filtered by feature
        ffPopulateStationSelect(stationSelect, feature);

        // Pre-select the specified station if provided
        if (preselect && stationSelect.querySelector(`option[value="${preselect}"]`)) {
            stationSelect.value = preselect;
        }

        stationSelect.addEventListener('change', () => {
            ffSyncStationDateBounds(row, feature);
        });

        removeBtn.addEventListener('click', () => {
            row.remove();
            ffUpdateRemoveButtons(container);
        });

        stationRowsContainer.appendChild(fragment);

        // Get the inserted row (last child)
        const insertedRow = stationRowsContainer.lastElementChild;
        ffSyncStationDateBounds(insertedRow, feature);
        ffUpdateRemoveButtons(container);
    }

    function ffPopulateStationSelect(selectEl, feature) {
        const prevValue = selectEl.value;
        selectEl.innerHTML = '';
        if (!feature) return;

        const ds = state.activeDatasetFilter;
        state.bootstrap.station_names.forEach(stationName => {
            const meta = state.stationsByName.get(stationName);
            if (!meta || !meta.features.includes(feature)) return;
            if (ds !== 'all' && meta.dataset !== ds) return;
            const option = document.createElement('option');
            option.value = stationName;
            option.textContent = prettyStation(stationName);
            if (stationName === prevValue) option.selected = true;
            selectEl.appendChild(option);
        });
    }

    function ffSyncStationOptions(row, feature) {
        const stationSelect = row.querySelector('.sr-station-select');
        ffPopulateStationSelect(stationSelect, feature);
        ffSyncStationDateBounds(row, feature);
    }

    function ffSyncStationDateBounds(row, feature) {
        const station = row.querySelector('.sr-station-select').value;
        const startInput = row.querySelector('.sr-start');
        const endInput = row.querySelector('.sr-end');
        const metaEl = row.querySelector('.station-row-meta');

        if (!feature) feature = els.seriesBuilder.querySelector('.ff-feature-select')?.value;

        const stationMeta = state.stationsByName.get(station);
        if (!stationMeta || !feature || !stationMeta.feature_details[feature]) {
            metaEl.textContent = 'Select a station to see available date range.';
            return;
        }

        const detail = stationMeta.feature_details[feature];
        startInput.min = detail.start_date;
        startInput.max = detail.end_date;
        endInput.min = detail.start_date;
        endInput.max = detail.end_date;
        if (!startInput.value || startInput.value < detail.start_date || startInput.value > detail.end_date) {
            startInput.value = detail.start_date;
        }
        if (!endInput.value || endInput.value > detail.end_date || endInput.value < detail.start_date) {
            endInput.value = detail.end_date;
        }
        if (startInput.value > endInput.value) {
            startInput.value = detail.start_date;
            endInput.value = detail.end_date;
        }

        metaEl.textContent = `Coverage: ${detail.start_date} → ${detail.end_date} · ${detail.observations.toLocaleString()} obs · Unit: ${detail.unit}`;
    }

    function ffUpdateRemoveButtons(container) {
        const rows = container.querySelectorAll('.station-row-item');
        const showDelete = rows.length > 1;
        rows.forEach(row => {
            const btn = row.querySelector('.station-row-remove-btn');
            if (btn) {
                if (showDelete) btn.classList.remove('hidden');
                else btn.classList.add('hidden');
            }
        });
    }

    async function submitSeriesRequest(mode) {
        const payload = {
            graph_type: els.graphTypeSelect.value,
            series: getSeriesSelections(),
        };
        const targetMessage = els.builderMessage;
        showMessage(targetMessage, 'Working on your request…', '');
        try {
            const endpoint = mode === 'analysis' ? '/api/analyze' : '/api/visualize';
            const response = await fetch(endpoint, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
            });
            const data = await response.json();
            if (!response.ok || !data.ok) {
                throw new Error(data.error || 'Request failed.');
            }
            if (mode === 'analysis') {
                appendAnalysisCard(data.result);
                activateDockTab('analysis');
                showMessage(targetMessage, 'Analysis card added.', 'success');
            } else {
                appendVisualizationCard(data.result);
                activateDockTab('visualize');
                showMessage(targetMessage, 'Visualization card added.', 'success');
            }
        } catch (error) {
            showMessage(targetMessage, error.message || 'Something went wrong.', 'error');
        }
    }

    async function runPrediction() {
        const payload = {
            station: els.predictStationSelect.value,
            feature: els.predictFeatureSelect.value,
            horizon: Number(els.predictHorizonInput.value || 0),
        };
        showMessage(els.predictionMessage, 'Running prediction…', '');
        try {
            const response = await fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
            });
            const data = await response.json();
            if (!response.ok || !data.ok) {
                throw new Error(data.error || 'Prediction failed.');
            }
            appendPredictionCard(data.result);
            activateDockTab('prediction');
            showMessage(els.predictionMessage, 'Prediction card added.', 'success');
        } catch (error) {
            showMessage(els.predictionMessage, error.message || 'Something went wrong.', 'error');
        }
    }

    function appendVisualizationCard(result) {
        clearEmptyStateIfNeeded(els.visualizationCards);
        const cardId = `visual-${++state.cardCounters.visualize}`;
        const card = buildBaseCard(cardId, result.title, describeSeries(result.series));
        els.visualizationCards.prepend(card);
        renderPlot(card.querySelector('.plot-container'), result.figure);
    }

    function appendAnalysisCard(result) {
        clearEmptyStateIfNeeded(els.analysisCards);
        const cardId = `analysis-${++state.cardCounters.analysis}`;
        const card = buildBaseCard(cardId, result.title, describeSeries(result.series));
        const body = card.querySelector('.workspace-card-body');
        const analysis = result.analysis;
        const analysisBlock = document.createElement('div');
        analysisBlock.className = 'analysis-block';
        analysisBlock.innerHTML = `
            <h4>🧠 AI Summary</h4>
            <div class="analysis-summary"></div>
            <div class="analysis-findings"></div>
            <div class="analysis-comparisons"></div>
        `;
        body.appendChild(analysisBlock);
        els.analysisCards.prepend(card);
        renderPlot(card.querySelector('.plot-container'), result.figure);
        analysisBlock.querySelector('.analysis-summary').innerHTML = analysis.summary;
        const findingsWrap = analysisBlock.querySelector('.analysis-findings');
        analysis.findings.forEach((finding) => {
            const findingEl = document.createElement('div');
            findingEl.className = 'finding-card';
            findingEl.innerHTML = `<div class="finding-title">${escapeHtml(finding.title)}</div><div>${escapeHtml(finding.body)}</div>`;
            findingsWrap.appendChild(findingEl);
        });
        const compareWrap = analysisBlock.querySelector('.analysis-comparisons');
        if (analysis.comparisons && analysis.comparisons.length) {
            const heading = document.createElement('h4');
            heading.textContent = 'Cross-series comparisons';
            compareWrap.appendChild(heading);
            analysis.comparisons.forEach((note) => {
                const noteEl = document.createElement('div');
                noteEl.className = 'compare-note';
                noteEl.textContent = note;
                compareWrap.appendChild(noteEl);
            });
        }
    }

    function appendPredictionCard(result) {
        clearEmptyStateIfNeeded(els.predictionCards);
        const subtitle = `${prettyStation(result.station)} · ${prettyFeature(result.feature)} · Horizon ${result.horizon} ${result.frequency === 'monthly' ? 'month(s)' : 'day(s)'}`;
        const cardId = `prediction-${++state.cardCounters.prediction}`;
        const card = buildBaseCard(cardId, result.title, subtitle);
        const body = card.querySelector('.workspace-card-body');
        // basePlot is the plot-container created by buildBaseCard (used for zoom — shown in fullscreen)
        const basePlot = body.querySelector('.plot-container');

        // Insert zoom label BEFORE basePlot so it appears at the top
        const zoomLabel = document.createElement('div');
        zoomLabel.className = 'chart-section-label';
        zoomLabel.textContent = 'Zoomed view · Last 1 year + forecast';
        body.insertBefore(zoomLabel, basePlot);

        // Full history section — appended after basePlot
        const fullLabel = document.createElement('div');
        fullLabel.className = 'chart-section-label';
        fullLabel.textContent = 'Full historical view';
        body.appendChild(fullLabel);

        const fullPlot = document.createElement('div');
        fullPlot.className = 'plot-container';
        body.appendChild(fullPlot);

        // AI analysis block
        const block = document.createElement('div');
        block.className = 'analysis-block';
        block.innerHTML = `<h4>🧠 Prediction Analysis</h4><div class="analysis-summary"></div>`;
        const summaryEl = block.querySelector('.analysis-summary');
        if (result.summary && result.summary.startsWith('🧠 Analysis:')) {
            summaryEl.innerHTML = result.summary.replace('🧠 Analysis:\n', '');
        } else {
            summaryEl.textContent = result.summary;
        }
        body.appendChild(block);

        // Prepend to DOM first so Plotly can measure container width correctly
        els.predictionCards.prepend(card);

        // Render zoom into basePlot (first .plot-container → used by openFullscreen)
        renderPlot(basePlot, result.figure_zoom || result.figure);
        // Render full history into the separate fullPlot div
        renderPlot(fullPlot, result.figure);
    }

    function buildBaseCard(cardId, title, subtitle) {
        const card = document.createElement('article');
        card.className = 'workspace-card';
        card.innerHTML = `
            <div class="workspace-card-header">
                <div style="flex: 1; min-width: 0;">
                    <h3 class="workspace-card-title">${escapeHtml(title)}</h3>
                    <div class="workspace-card-subtitle">${escapeHtml(subtitle)}</div>
                </div>
                <div class="card-header-actions" style="display: flex; gap: 8px; flex-shrink: 0; align-items: start;">
                    <button class="expand-btn ghost-btn icon-only" type="button" title="Expand to fullscreen">
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="15 3 21 3 21 9"/><polyline points="9 21 3 21 3 15"/><line x1="21" y1="3" x2="14" y2="10"/><line x1="3" y1="21" x2="10" y2="14"/></svg>
                    </button>
                    <button class="delete-btn ghost-btn" type="button">Delete</button>
                </div>
            </div>
            <div class="workspace-card-body">
                <div class="plot-container" id="${cardId}"></div>
            </div>
        `;
        card.querySelector('.delete-btn').addEventListener('click', () => {
            const parent = card.parentElement;
            card.remove();
            if (!parent.children.length) {
                if (parent === els.visualizationCards) {
                    setEmptyState(els.visualizationCards, 'No visualizations yet. Use the Explore section on the left to add your first chart.');
                } else if (parent === els.analysisCards) {
                    setEmptyState(els.analysisCards, 'No analysis cards yet. Build a selection and choose “Add to analyse”.');
                } else {
                    setEmptyState(els.predictionCards, 'No predictions yet. Configure a station, feature, and horizon on the left.');
                }
            }
        });
        card.querySelector('.expand-btn').addEventListener('click', () => openFullscreen(card));
        return card;
    }

    function openFullscreen(card) {
        const title = card.querySelector('.workspace-card-title').textContent;
        const subtitle = card.querySelector('.workspace-card-subtitle').textContent;
        const bodyContent = card.querySelector('.workspace-card-body').cloneNode(true);

        els.modalTitle.textContent = title;
        els.modalSubtitle.textContent = subtitle;
        els.modalBody.innerHTML = '';
        els.modalBody.appendChild(bodyContent);

        els.fullscreenModal.classList.remove('hidden');
        document.body.classList.add('modal-open');

        // Re-render every plot container in the modal so all charts expand to modal size
        bodyContent.querySelectorAll('.plot-container').forEach((plot, i) => {
            plot.id = `modal-plot-container-${i}`;
            if (plot.dataset.figure) {
                renderPlot(plot, plot.dataset.figure);
            }
        });
    }

    function closeFullscreen() {
        els.fullscreenModal.classList.add('hidden');
        document.body.classList.remove('modal-open');
        els.modalBody.innerHTML = '';
    }

    function toggleMapFullscreen() {
        const stage = document.querySelector('.map-stage');
        const btn = els.mapFullscreenToggle;
        const isNowFull = stage.classList.toggle('fullscreen');
        document.body.classList.toggle('map-fullscreen');
        
        if (isNowFull) {
            btn.innerHTML = `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="4 14 10 14 10 20"/><polyline points="20 10 14 10 14 4"/><line x1="14" y1="10" x2="21" y2="3"/><line x1="10" y1="14" x2="3" y2="21"/></svg>`;
            btn.title = "Exit fullscreen map";
        } else {
            btn.innerHTML = `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="15 3 21 3 21 9"/><polyline points="9 21 3 21 3 15"/><line x1="21" y1="3" x2="14" y2="10"/><line x1="3" y1="21" x2="10" y2="14"/></svg>`;
            btn.title = "Toggle fullscreen map";
        }

        // Leaflet needs to know the container size changed
        setTimeout(() => {
            if (state.map) {
                state.map.invalidateSize({ animate: true });
            }
        }, 300);
    }

    function stripPlotlyUids(obj) {
        if (Array.isArray(obj)) {
            obj.forEach(stripPlotlyUids);
        } else if (typeof obj === 'object' && obj !== null) {
            if ('uid' in obj) {
                delete obj.uid;
            }
            for (const key of Object.keys(obj)) {
                stripPlotlyUids(obj[key]);
            }
        }
    }

    function renderPlot(container, figureJson) {
        container.dataset.figure = figureJson;
        // Defer to next animation frame so the browser finishes layout before
        // Plotly measures the container width (fixes clipping in card view).
        requestAnimationFrame(() => {
            const figure = JSON.parse(figureJson);
            stripPlotlyUids(figure);
            const config = {
                responsive: true,
                displaylogo: false,
                modeBarButtonsToRemove: ['lasso2d', 'select2d', 'autoScale2d'],
            };
            Plotly.newPlot(container, figure.data, figure.layout, config);
        });
    }

    function focusStationFromSearch() {
        const query = (els.stationSearch.value || '').trim();
        if (!query) {
            showMessage(els.builderMessage, 'Enter a station name to focus it on the map.', 'error');
            return;
        }
        const prettyQuery = query;
        const exact = state.bootstrap.station_names.find((station) => prettyStation(station).toLowerCase() === prettyQuery.toLowerCase() || station.toLowerCase() === prettyQuery.toLowerCase());
        const partial = state.bootstrap.station_names.find((station) => prettyStation(station).toLowerCase().includes(prettyQuery.toLowerCase()) || station.toLowerCase().includes(prettyQuery.toLowerCase()));
        const station = exact || partial;
        if (!station) {
            showMessage(els.builderMessage, `No station matched "${query}".`, 'error');
            return;
        }
        focusStation(station, true);
        els.mapSearchBar.classList.add('hidden');
        els.clearStationFocusBtn.classList.remove('hidden');
        showMessage(els.builderMessage, `${prettyStation(station)} highlighted on the map.`, 'success');
    }

    function focusStation(stationName, openCard) {
        state.focusedStation = stationName;
        const marker = state.markers.get(stationName);
        const stationMeta = state.stationsByName.get(stationName);
        if (!marker || !stationMeta) return;
        refreshMarkerVisibility();
        state.map.setView([stationMeta.lat, stationMeta.lon], Math.max(state.map.getZoom(), 7), { animate: true });
        marker.openTooltip();
        if (openCard) {
            populateStationCard(stationMeta);
            els.mapSearchBar.classList.add('hidden');
        }
        seedBuilderFromStation(stationName);
    }

    function clearStationFocus() {
        state.focusedStation = null;
        els.stationSearch.value = '';
        els.clearStationFocusBtn.classList.add('hidden');
        els.mapSearchBar.classList.remove('hidden');
        refreshMarkerVisibility();
        els.stationDetailCard.classList.add('hidden');
        showMessage(els.builderMessage, '', '');
    }

    function populateStationCard(stationMeta) {
        els.detailStationName.textContent = prettyStation(stationMeta.station);
        els.stationMetaGrid.innerHTML = '';
        els.stationFeatureBadges.innerHTML = '';
        const detailPairs = [
            ['Country', stationMeta.country],
            ['Latitude', String(stationMeta.lat)],
            ['Longitude', String(stationMeta.lon)],
            ['Latest timestamp', stationMeta.latest_timestamp || '—'],
        ];
        detailPairs.forEach(([label, value]) => {
            const item = document.createElement('div');
            item.className = 'meta-item';
            item.innerHTML = `<div class="meta-label">${escapeHtml(label)}</div><div class="meta-value">${escapeHtml(value)}</div>`;
            els.stationMetaGrid.appendChild(item);
        });
        stationMeta.features.forEach((feature) => {
            const detail = stationMeta.feature_details[feature];
            const badge = document.createElement('div');
            badge.className = 'feature-badge';
            badge.textContent = `${prettyFeature(feature)} · ${detail.start_date} → ${detail.end_date}`;
            els.stationFeatureBadges.appendChild(badge);
        });
        els.stationDetailCard.classList.remove('hidden');
    }

    function seedBuilderFromStation(stationName) {
        const rows = Array.from(els.seriesBuilder.querySelectorAll('.series-row'));
        if (!rows.length) return;
        const firstRow = rows[0];
        firstRow.querySelector('.series-station').value = stationName;
        syncRowFeatureOptions(firstRow);
    }

    function refreshMarkerVisibility() {
        state.bootstrap.stations.forEach((station) => {
            const marker = state.markers.get(station.station);
            if (!marker) return;
            const passesFeature = state.activeFeatureFilter === 'All' || station.features.includes(state.activeFeatureFilter);
            const passesDataset = state.activeDatasetFilter === 'all' || station.dataset === state.activeDatasetFilter;
            if (passesFeature && passesDataset) {
                if (!state.map.hasLayer(marker)) marker.addTo(state.map);
                marker.setStyle(markerStyleForStation(station.station));
            } else if (state.map.hasLayer(marker)) {
                state.map.removeLayer(marker);
            }
        });
    }

    // Mekong: colors by feature count (filled, opaque)
    const MEKONG_COLORS = { 1: '#60a5fa', 2: '#a78bfa', 3: '#34d399', 4: '#f59e0b' };
    // LamaH: orange palette (semi-transparent, thinner border)
    const LAMAH_COLORS = { 1: '#fb923c', 2: '#f472b6', 3: '#34d399', 4: '#facc15' };

    function markerStyleForStation(stationName) {
        const station = state.stationsByName.get(stationName);
        const isFocused = state.focusedStation === stationName;
        const featureCount = station?.features?.length || 1;
        const isLamaH = station?.dataset === 'lamah';
        if (isFocused) {
            return { radius: 9, color: '#ef4444', weight: 2.2, fillColor: '#ef4444', fillOpacity: 0.95 };
        }
        if (isLamaH) {
            return {
                radius: 3,
                color: '#f54842', //'#94a3b8',
                weight: 0.5,
                fillColor: LAMAH_COLORS[featureCount] || '#f54842',
                fillOpacity: 0.5,
            };
        }
        return {
            radius: 6,
            color: '#e2e8f0',
            weight: 1.2,
            fillColor: MEKONG_COLORS[featureCount] || '#60a5fa',
            fillOpacity: 0.82,
        };
    }

    function buildTooltip(station) {
        const featureText = station.features.map(prettyFeature).join(', ') || 'None';
        const displayName = station.name && station.name !== station.station
            ? escapeHtml(station.name)
            : escapeHtml(prettyStation(station.station));
        const datasetLabel = station.dataset === 'lamah' ? 'LamaH-CE' : 'Mekong';
        return `
            <div class="station-tooltip-content">
                <div class="station-tooltip-title">${displayName}</div>
                <div class="station-tooltip-meta">${escapeHtml(station.country)} · ${datasetLabel}</div>
                <div class="station-tooltip-meta">Features: ${escapeHtml(featureText)}</div>
            </div>
        `;
    }

    function activateDockTab(tabName) {
        els.dockTabs.forEach((tab) => tab.classList.toggle('active', tab.dataset.dockTab === tabName));
        els.workspacePills.forEach((pill) => pill.classList.toggle('active', pill.dataset.tabTarget === tabName));
        els.dockPanels.forEach((panel) => panel.classList.toggle('active', panel.dataset.dockPanel === tabName));
        els.dockClearBtns.forEach((btn) => btn.classList.toggle('hidden', btn.dataset.dockClear !== tabName));

        // Toggle sidebar panels
        document.querySelectorAll('[data-sidebar-panel]').forEach(panel => {
            const key = panel.dataset.sidebarPanel;
            if (key === 'visualize') {
                // Show explore/analyse setup for both visualize and analysis modes
                panel.classList.toggle('hidden', tabName === 'prediction');
            } else if (key === 'prediction') {
                panel.classList.toggle('hidden', tabName !== 'prediction');
            }
        });

        // Toggle action buttons within the explore panel
        document.querySelectorAll('[data-sidebar-action]').forEach(row => {
            const key = row.dataset.sidebarAction;
            if (key === 'visualize') {
                row.classList.toggle('hidden', tabName !== 'visualize');
            } else if (key === 'analysis') {
                row.classList.toggle('hidden', tabName !== 'analysis');
            }
        });
    }

    function initDockResizer() {
        let resizing = false;
        let startY = 0;
        let startHeight = 0;

        const onMove = (event) => {
            if (!resizing) return;
            const delta = startY - event.clientY;
            const nextHeight = Math.max(120, Math.min(window.innerHeight - 200, startHeight + delta));
            els.dock.style.height = `${nextHeight}px`;
            window.dispatchEvent(new Event('resize'));
        };

        const onUp = () => {
            resizing = false;
            document.body.style.userSelect = '';
            window.removeEventListener('mousemove', onMove);
            window.removeEventListener('mouseup', onUp);
        };

        els.dockHandle.addEventListener('mousedown', (event) => {
            resizing = true;
            startY = event.clientY;
            startHeight = els.dock.getBoundingClientRect().height;
            document.body.style.userSelect = 'none';
            window.addEventListener('mousemove', onMove);
            window.addEventListener('mouseup', onUp);
        });
    }

    function setEmptyState(container, message) {
        container.innerHTML = `<div class="empty-state">${escapeHtml(message)}</div>`;
    }

    function clearEmptyStateIfNeeded(container) {
        if (container.querySelector('.empty-state')) {
            container.innerHTML = '';
        }
    }

    function describeSeries(series) {
        return series.map((item) => `${prettyStation(item.station)} · ${prettyFeature(item.feature)} · ${item.start_date} → ${item.end_date}`).join('\n');
    }

    function showMessage(element, text, type) {
        element.textContent = text;
        element.className = `inline-message${type ? ` ${type}` : ''}`;
    }

    function prettyStation(name) {
        return name.replaceAll('_', ' ');
    }

    function prettyFeature(feature) {
        return feature.replaceAll('_', ' ');
    }

    function formatNumber(value) {
        return typeof value === 'number' && Number.isFinite(value) ? value.toFixed(2) : '—';
    }

    function escapeHtml(value) {
        return String(value)
            .replaceAll('&', '&amp;')
            .replaceAll('<', '&lt;')
            .replaceAll('>', '&gt;')
            .replaceAll('"', '&quot;')
            .replaceAll("'", '&#039;');
    }
})();
