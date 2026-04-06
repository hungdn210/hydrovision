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
        predictDatasetFilter: 'all',
        predictMode: 'future',
        predictStationsForModel: {
            lamah: { historical: new Set(), future: new Set() },
            mekong: { historical: new Set(), future: new Set() },
        },
        analysisMode: 'free',
        focusedStation: null,
        compareDataset: 'mekong',
        cardCounters: {
            visualize: 0,
            analysis: 0,
            prediction: 0,
            climate: 0,
            changepoint: 0,
            animate: 0,
            modelcompare: 0,
            decompose: 0,
            wavelet: 0,
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
        addSeriesRowBtn: document.getElementById('addSeriesRowBtn'),
        addVisualizationBtn: document.getElementById('addVisualizationBtn'),
        addAnalysisBtn: document.getElementById('addAnalysisBtn'),
        builderMessage: document.getElementById('builderMessage'),
        analysisModeSwitch: document.getElementById('analysisModeSwitch'),
        freeDatasetPicker: document.getElementById('freeDatasetPicker'),
        freeAnalysisBuilder: document.getElementById('freeAnalysisBuilder'),
        addFreeSeriesRowBtn: document.getElementById('addFreeSeriesRowBtn'),
        runFreeAnalysisBtn: document.getElementById('runFreeAnalysisBtn'),
        freeAnalysisMessage: document.getElementById('freeAnalysisMessage'),
        datasetPicker: document.getElementById('datasetPicker'),
        flyToMekong: document.getElementById('flyToMekong'),
        flyToLamaH: document.getElementById('flyToLamaH'),
        stationDetailCard: document.getElementById('stationDetailCard'),
        detailStationName: document.getElementById('detailStationName'),
        stationMetaGrid: document.getElementById('stationMetaGrid'),
        stationFeatureBadges: document.getElementById('stationFeatureBadges'),
        stationIndicesSection: document.getElementById('stationIndicesSection'),
        closeStationCardBtn: document.getElementById('closeStationCardBtn'),
        dock: document.getElementById('dock'),
        dockHandle: document.getElementById('dockHandle'),
        dockTabs: Array.from(document.querySelectorAll('.dock-tab')),
        dockPanels: Array.from(document.querySelectorAll('.dock-panel')),
        dockClearBtns: Array.from(document.querySelectorAll('.dock-clear-btn')),
        visualizationCards: document.getElementById('visualizationCards'),
        analysisCards: document.getElementById('analysisCards'),
        predictionCards: document.getElementById('predictionCards'),
        clearVisualizationsBtn: document.getElementById('clearVisualizationsBtn'),
        clearAnalysisBtn: document.getElementById('clearAnalysisBtn'),
        clearPredictionBtn: document.getElementById('clearPredictionBtn'),
        clearCompareBtn: document.getElementById('clearCompareBtn'),
        compareDatasetPicker: document.getElementById('compareDatasetPicker'),
        compareFeatureSelect: document.getElementById('compareFeatureSelect'),
        compareYearInput: document.getElementById('compareYearInput'),
        compareYearHint: document.getElementById('compareYearHint'),
        compareComponentSelect: document.getElementById('compareComponentSelect'),
        runCompareBtn: document.getElementById('runCompareBtn'),
        compareMessage: document.getElementById('compareMessage'),
        compareWorkspace: document.getElementById('compareWorkspace'),
        // Quality
        qualityWorkspace: document.getElementById('qualityWorkspace'),
        clearQualityBtn: document.getElementById('clearQualityBtn'),
        qualityViewPicker: document.getElementById('qualityViewPicker'),
        qualityStationGroup: document.getElementById('qualityStationGroup'),
        qualityDatasetGroup: document.getElementById('qualityDatasetGroup'),
        qualityZGroup: document.getElementById('qualityZGroup'),
        qualityStationSelect: document.getElementById('qualityStationSelect'),
        qualityFeatureSelect: document.getElementById('qualityFeatureSelect'),
        qualityDatasetPicker: document.getElementById('qualityDatasetPicker'),
        qualityImpFeatureSelect: document.getElementById('qualityImpFeatureSelect'),
        qualityZSlider: document.getElementById('qualityZSlider'),
        qualityZDisplay: document.getElementById('qualityZDisplay'),
        runQualityBtn: document.getElementById('runQualityBtn'),
        qualityMessage: document.getElementById('qualityMessage'),
        // Scenario
        scenarioCards: document.getElementById('scenarioCards'),
        clearScenarioBtn: document.getElementById('clearScenarioBtn'),
        scenarioStationSelect: document.getElementById('scenarioStationSelect'),
        scenarioTargetSelect: document.getElementById('scenarioTargetSelect'),
        scenarioDriverSelect: document.getElementById('scenarioDriverSelect'),
        scenarioModelSelect: document.getElementById('scenarioModelSelect'),
        scenarioScaleSlider: document.getElementById('scenarioScaleSlider'),
        scenarioScaleDisplay: document.getElementById('scenarioScaleDisplay'),
        scenarioDurationSlider: document.getElementById('scenarioDurationSlider'),
        scenarioDurationDisplay: document.getElementById('scenarioDurationDisplay'),
        scenarioOffsetSlider: document.getElementById('scenarioOffsetSlider'),
        scenarioOffsetDisplay: document.getElementById('scenarioOffsetDisplay'),
        runScenarioBtn: document.getElementById('runScenarioBtn'),
        scenarioMessage: document.getElementById('scenarioMessage'),
        // Extreme Events
        extremeCards: document.getElementById('extremeCards'),
        clearExtremeBtn: document.getElementById('clearExtremeBtn'),
        extremeStationSelect: document.getElementById('extremeStationSelect'),
        extremeFeatureSelect: document.getElementById('extremeFeatureSelect'),
        extremeDistSelect: document.getElementById('extremeDistSelect'),
        runExtremeBtn: document.getElementById('runExtremeBtn'),
        extremeMessage: document.getElementById('extremeMessage'),
        // Risk Map
        riskWorkspace: document.getElementById('riskWorkspace'),
        clearRiskBtn: document.getElementById('clearRiskBtn'),
        riskDatasetSelect: document.getElementById('riskDatasetSelect'),
        riskFeatureSelect: document.getElementById('riskFeatureSelect'),
        riskLookbackSlider: document.getElementById('riskLookbackSlider'),
        riskLookbackDisplay: document.getElementById('riskLookbackDisplay'),
        runRiskBtn: document.getElementById('runRiskBtn'),
        riskMessage: document.getElementById('riskMessage'),
        // Climate Projector
        climateCards: document.getElementById('climateCards'),
        clearClimateBtn: document.getElementById('clearClimateBtn'),
        climateDatasetSelect: document.getElementById('climateDatasetSelect'),
        climateStationSelect: document.getElementById('climateStationSelect'),
        climateFeatureSelect: document.getElementById('climateFeatureSelect'),
        climateYearsSlider: document.getElementById('climateYearsSlider'),
        climateYearsDisplay: document.getElementById('climateYearsDisplay'),
        runClimateBtn: document.getElementById('runClimateBtn'),
        climateMessage: document.getElementById('climateMessage'),
        // Change Point Detection
        changepointCards: document.getElementById('changepointCards'),
        clearChangepointBtn: document.getElementById('clearChangepointBtn'),
        cpDatasetSelect: document.getElementById('cpDatasetSelect'),
        cpStationSelect: document.getElementById('cpStationSelect'),
        cpFeatureSelect: document.getElementById('cpFeatureSelect'),
        cpMethodPicker: document.getElementById('cpMethodPicker'),
        cpBreaksSlider: document.getElementById('cpBreaksSlider'),
        cpBreaksDisplay: document.getElementById('cpBreaksDisplay'),
        runCpBtn: document.getElementById('runCpBtn'),
        cpMessage: document.getElementById('cpMessage'),
        // Animated Map
        animateCards: document.getElementById('animateCards'),
        clearAnimateBtn: document.getElementById('clearAnimateBtn'),
        animateDatasetSelect: document.getElementById('animateDatasetSelect'),
        animateFeatureSelect: document.getElementById('animateFeatureSelect'),
        runAnimateBtn: document.getElementById('runAnimateBtn'),
        animateMessage: document.getElementById('animateMessage'),
        // Model Comparison
        mcCards: document.getElementById('mcCards'),
        clearMcBtn: document.getElementById('clearMcBtn'),
        mcDatasetSelect: document.getElementById('mcDatasetSelect'),
        mcStationSelect: document.getElementById('mcStationSelect'),
        mcFeatureSelect: document.getElementById('mcFeatureSelect'),
        mcHorizonSlider: document.getElementById('mcHorizonSlider'),
        mcHorizonDisplay: document.getElementById('mcHorizonDisplay'),
        runMcBtn: document.getElementById('runMcBtn'),
        mcMessage: document.getElementById('mcMessage'),
        // STL Decomposition
        decompCards: document.getElementById('decompCards'),
        clearDecompBtn: document.getElementById('clearDecompBtn'),
        decompDatasetSelect: document.getElementById('decompDatasetSelect'),
        decompStationSelect: document.getElementById('decompStationSelect'),
        decompFeatureSelect: document.getElementById('decompFeatureSelect'),
        runDecompBtn: document.getElementById('runDecompBtn'),
        decompMessage: document.getElementById('decompMessage'),
        // Wavelet Analysis
        waveletCards: document.getElementById('waveletCards'),
        clearWaveletBtn: document.getElementById('clearWaveletBtn'),
        waveletDatasetSelect: document.getElementById('waveletDatasetSelect'),
        waveletStationSelect: document.getElementById('waveletStationSelect'),
        waveletFeatureSelect: document.getElementById('waveletFeatureSelect'),
        runWaveletBtn: document.getElementById('runWaveletBtn'),
        waveletMessage: document.getElementById('waveletMessage'),
        // Network
        clearNetworkBtn: document.getElementById('clearNetworkBtn'),
        runNetworkBtn: document.getElementById('runNetworkBtn'),
        networkMessage: document.getElementById('networkMessage'),
        networkWorkspace: document.getElementById('networkWorkspace'),
        networkViewPicker: document.getElementById('networkViewPicker'),
        networkContribGroup: document.getElementById('networkContribGroup'),
        networkContribStation: document.getElementById('networkContribStation'),
        netChipNodes: document.getElementById('netChipNodes'),
        netChipEdges: document.getElementById('netChipEdges'),
        netChipStem: document.getElementById('netChipStem'),
        exportPdfBtn: document.getElementById('exportPdfBtn'),
        predictModePicker: document.getElementById('predictModePicker'),
        predictHorizonLabel: document.getElementById('predictHorizonLabel'),
        predictDatasetPicker: document.getElementById('predictDatasetPicker'),
        predictStationSelect: document.getElementById('predictStationSelect'),
        predictFeatureSelect: document.getElementById('predictFeatureSelect'),
        predictModelSelect: document.getElementById('predictModelSelect'),
        predictHorizonInput: document.getElementById('predictHorizonInput'),
        predictAnalysisToggle: document.getElementById('predictAnalysisToggle'),
        predictCIToggle: document.getElementById('predictCIToggle'),
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
        'Seasonal Subseries Plot': {
            desc: 'Draws 12 mini time-series — one per calendar month — each showing that month\'s values across all years. Trend lines (red = declining, green = rising) make long-term shifts within a season immediately visible.',
            rules: 'Exactly 1 row. Single station and feature required.',
            allowMultiFeature: false,
            allowMultiRow: false
        },
        'Calendar Heatmap': {
            desc: 'A grid with years on the Y-axis and months on the X-axis. Each cell is coloured by the monthly mean value — darker = higher. Instantly reveals wet/dry seasons and anomalous years across decades.',
            rules: 'Exactly 1 row. Single station and feature required.',
            allowMultiFeature: false,
            allowMultiRow: false
        },
        'Rolling Correlation Chart': {
            desc: 'Plots the Pearson correlation between two features in a sliding time window (365-day for daily data, 12-month for monthly). Shows whether the relationship between variables is stable year-round or varies seasonally.',
            rules: 'Select exactly 2 features from the same station.',
            allowMultiFeature: true,
            allowMultiRow: false
        },
        'Exceedance Probability Curve': {
            desc: 'Like a Flow Duration Curve but on a log-probability x-axis — the standard format for flood frequency analysis. Return period markers (2yr, 5yr, 10yr…) show the magnitude of rare events.',
            rules: 'Exactly 1 row. Single station and feature required.',
            allowMultiFeature: false,
            allowMultiRow: false
        },
        'Granger Causality': {
            desc: 'A statistical test for determining whether one hydrological time series is useful in forecasting another. It reveals directional "causality" (e.g., does upstream rainfall actually cause downstream level changes).',
            rules: 'Select exactly 2 features from the same station. Use the multi-select to pick both.',
            allowMultiFeature: true,
            allowMultiRow: false
        },
        'Cross-Correlation Function (CCF)': {
            desc: 'Analyzes the correlation between two series at different time lags. Essential for determining the "travel time" of water between upstream and downstream stations.',
            rules: 'Exactly 2 rows required. Can use different stations or different features.',
            allowMultiFeature: false,
            allowMultiRow: true
        },
        'STL Decomposition': {
            desc: 'Decomposes a time series into Trend, Seasonal, and Residual (noise) components. Visualizes the underlying long-term signal without seasonal "noise".',
            rules: 'Exactly 1 row. Single station and feature required.',
            allowMultiFeature: false,
            allowMultiRow: false
        },
        'Change-Point Detection': {
            desc: 'Statistically identifies abrupt shifts in the mean or variance of a series. Useful for detecting the impact of dam construction or major land-use changes.',
            rules: 'Exactly 1 row. Single station and feature required.',
            allowMultiFeature: false,
            allowMultiRow: false
        },
        'Wavelet Analysis': {
            desc: 'A powerful tool for analyzing localized variations of power within a time series. Shows how periodic signals (like the annual cycle) change in strength over decades.',
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
        setEmptyState(els.analysisCards, 'No analysis cards yet. Build a selection and choose “Add to analyze”.');
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
        await populateModelDropdown();
        await fetchPredictStations(els.predictModelSelect?.value || 'FlowNet');
        populatePredictionControls();
        populateScenarioControls();
        setEmptyState(els.scenarioCards, 'No scenarios yet. Configure a station and driver above, then click Run scenario.');
        populateQualityControls();
        populateNetworkContribFromBootstrap();
        populateExtremeControls();
        setEmptyState(els.extremeCards, 'No analyses yet. Select a station and feature, then click Run analysis.');
        populateRiskControls();
        initClimateControls();
        initChangepointControls();
        initAnimateControls();
        initModelCompareControls();
        initDecomposeControls();
        initWaveletControls();
        addSeriesRow();
        addFreeSeriesRow();
        syncSeriesBuilderUI();
        refreshMarkerVisibility();
        // Show Visualize tab by default
        activateDockTab('visualize');
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
                invalidateMapAfterTransition();
                activateDockTab(tab);
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
        els.addFreeSeriesRowBtn.addEventListener('click', () => addFreeSeriesRow());
        els.runFreeAnalysisBtn.addEventListener('click', submitFreeAnalysis);
        els.analysisModeSwitch.querySelectorAll('.mode-seg-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                state.analysisMode = btn.dataset.mode;
                applyAnalysisMode();
            });
        });
        els.clearVisualizationsBtn.addEventListener('click', () => setEmptyState(els.visualizationCards, 'No visualizations yet. Use the Explore section on the left to add your first chart.'));
        els.clearAnalysisBtn.addEventListener('click', () => setEmptyState(els.analysisCards, 'No analysis cards yet. Build a selection and choose “Add to analyze”.'));
        els.clearPredictionBtn.addEventListener('click', () => setEmptyState(els.predictionCards, 'No predictions yet. Configure a station, feature, and horizon on the left.'));
        els.clearCompareBtn?.addEventListener('click', () => {
            if (els.compareWorkspace) els.compareWorkspace.innerHTML = '';
        });
        els.runCompareBtn?.addEventListener('click', runComparison);
        els.exportPdfBtn?.addEventListener('click', generateSessionPDF);

        // Quality
        els.clearQualityBtn?.addEventListener('click', () => {
            if (els.qualityWorkspace) els.qualityWorkspace.innerHTML = '';
        });
        els.runQualityBtn?.addEventListener('click', runQualityAnalysis);
        els.qualityZSlider?.addEventListener('input', () => {
            if (els.qualityZDisplay) els.qualityZDisplay.textContent = Number(els.qualityZSlider.value).toFixed(1);
        });
        els.qualityViewPicker?.querySelectorAll('.mode-seg-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                els.qualityViewPicker.querySelectorAll('.mode-seg-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                updateQualityControls(btn.dataset.view);
            });
        });
        els.qualityStationSelect?.addEventListener('change', updateQualityFeatureOptions);

        // Scenario
        els.clearScenarioBtn?.addEventListener('click', () =>
            setEmptyState(els.scenarioCards, 'No scenarios yet. Configure a station and driver above, then click Run scenario.')
        );
        els.scenarioStationSelect?.addEventListener('change', updateScenarioFeatureOptions);
        els.scenarioScaleSlider?.addEventListener('input', () => {
            const v = Number(els.scenarioScaleSlider.value);
            els.scenarioScaleDisplay.textContent = (v >= 0 ? '+' : '') + v + '%';
        });
        els.scenarioDurationSlider?.addEventListener('input', () => {
            const v = Number(els.scenarioDurationSlider.value);
            els.scenarioDurationDisplay.textContent = v + (v === 1 ? ' month' : ' months');
        });
        els.scenarioOffsetSlider?.addEventListener('input', () => {
            const v = Number(els.scenarioOffsetSlider.value);
            els.scenarioOffsetDisplay.textContent = 'month ' + (v + 1);
        });
        els.runScenarioBtn?.addEventListener('click', runScenario);

        // Extreme Events
        els.clearExtremeBtn?.addEventListener('click', () =>
            setEmptyState(els.extremeCards, 'No analyses yet. Select a station and feature, then click Run analysis.')
        );
        els.extremeStationSelect?.addEventListener('change', updateExtremeFeatureOptions);
        els.runExtremeBtn?.addEventListener('click', runExtremeAnalysis);

        // Risk Map
        els.clearRiskBtn?.addEventListener('click', () => {
            if (els.riskWorkspace) els.riskWorkspace.innerHTML = '';
        });
        els.riskDatasetSelect?.addEventListener('change', updateRiskFeatureOptions);
        els.riskLookbackSlider?.addEventListener('input', () => {
            if (els.riskLookbackDisplay) els.riskLookbackDisplay.textContent = els.riskLookbackSlider.value + ' pts';
        });
        els.runRiskBtn?.addEventListener('click', runRiskMap);

        // Climate Projector
        els.clearClimateBtn?.addEventListener('click', () => setEmptyState(els.climateCards, 'No projections yet.'));
        els.climateDatasetSelect?.addEventListener('change', () => updateSelectOptions(els.climateDatasetSelect.value, els.climateStationSelect, els.climateFeatureSelect));
        els.climateStationSelect?.addEventListener('change', () => updateFeatureSelectForStation(els.climateDatasetSelect.value, els.climateStationSelect.value, els.climateFeatureSelect));
        els.climateYearsSlider?.addEventListener('input', () => {
            if (els.climateYearsDisplay) els.climateYearsDisplay.textContent = els.climateYearsSlider.value + ' yrs';
        });
        els.runClimateBtn?.addEventListener('click', runClimateProjection);

        // Change Point Detection
        els.clearChangepointBtn?.addEventListener('click', () => setEmptyState(els.changepointCards, 'No analyses yet.'));
        els.cpDatasetSelect?.addEventListener('change', () => updateSelectOptions(els.cpDatasetSelect.value, els.cpStationSelect, els.cpFeatureSelect));
        els.cpStationSelect?.addEventListener('change', () => updateFeatureSelectForStation(els.cpDatasetSelect.value, els.cpStationSelect.value, els.cpFeatureSelect));
        els.cpBreaksSlider?.addEventListener('input', () => {
            if (els.cpBreaksDisplay) els.cpBreaksDisplay.textContent = els.cpBreaksSlider.value;
        });
        els.cpMethodPicker?.querySelectorAll('.mode-seg-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                els.cpMethodPicker.querySelectorAll('.mode-seg-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
            });
        });
        els.runCpBtn?.addEventListener('click', runChangePointDetection);

        // Animated Map
        els.clearAnimateBtn?.addEventListener('click', () => setEmptyState(els.animateCards, 'No animations yet.'));
        els.animateDatasetSelect?.addEventListener('change', () => updateAnimateFeatureOptions());
        els.runAnimateBtn?.addEventListener('click', runAnimatedMap);
        const animSpeedSlider = document.getElementById('animateSpeedSlider');
        const animSpeedDisplay = document.getElementById('animateSpeedDisplay');
        const speedLabels = ['0.5×', '1×', '2×', '4×', '8×'];
        animSpeedSlider?.addEventListener('input', () => {
            if (animSpeedDisplay) animSpeedDisplay.textContent = speedLabels[Number(animSpeedSlider.value) - 1];
        });

        // Model Comparison
        els.clearMcBtn?.addEventListener('click', () => setEmptyState(els.mcCards, 'No comparisons yet.'));
        els.mcDatasetSelect?.addEventListener('change', () => updateModelCompareOptions());
        els.mcStationSelect?.addEventListener('change', () => updateFeatureSelectForStation(els.mcDatasetSelect.value, els.mcStationSelect.value, els.mcFeatureSelect));
        els.mcFeatureSelect?.addEventListener('change', () => updateModelCompareOptions());
        els.mcHorizonSlider?.addEventListener('input', () => {
            if (els.mcHorizonDisplay) els.mcHorizonDisplay.textContent = els.mcHorizonSlider.value + ' days';
        });
        els.runMcBtn?.addEventListener('click', runModelComparison);

        // STL Decomposition
        els.clearDecompBtn?.addEventListener('click', () => setEmptyState(els.decompCards, 'No decompositions yet.'));
        els.decompDatasetSelect?.addEventListener('change', () => updateSelectOptions(els.decompDatasetSelect.value, els.decompStationSelect, els.decompFeatureSelect));
        els.decompStationSelect?.addEventListener('change', () => updateFeatureSelectForStation(els.decompDatasetSelect.value, els.decompStationSelect.value, els.decompFeatureSelect));
        els.runDecompBtn?.addEventListener('click', runDecomposition);

        // Wavelet Analysis
        els.clearWaveletBtn?.addEventListener('click', () => setEmptyState(els.waveletCards, 'No analyses yet.'));
        els.waveletDatasetSelect?.addEventListener('change', () => updateSelectOptions(els.waveletDatasetSelect.value, els.waveletStationSelect, els.waveletFeatureSelect));
        els.waveletStationSelect?.addEventListener('change', () => updateFeatureSelectForStation(els.waveletDatasetSelect.value, els.waveletStationSelect.value, els.waveletFeatureSelect));
        els.runWaveletBtn?.addEventListener('click', runWaveletAnalysis);

        // Network
        els.clearNetworkBtn?.addEventListener('click', () => {
            if (els.networkWorkspace) els.networkWorkspace.innerHTML = '';
        });
        els.runNetworkBtn?.addEventListener('click', runNetworkAnalysis);
        els.networkViewPicker?.querySelectorAll('.mode-seg-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                els.networkViewPicker.querySelectorAll('.mode-seg-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                const isContrib = btn.dataset.view === 'contribution';
                els.networkContribGroup?.classList.toggle('hidden', !isContrib);
            });
        });

        els.dockTabs.forEach((tab) => {
            tab.addEventListener('click', () => activateDockTab(tab.dataset.dockTab));
        });

        els.predictStationSelect.addEventListener('change', () => {
            updatePredictionFeatureOptions();
            checkPredictionCapability();
        });
        els.predictFeatureSelect.addEventListener('change', () => {
            updatePredictionHint();
            checkPredictionCapability();
        });
        els.runPredictionBtn.addEventListener('click', runPrediction);
        els.predictModelSelect?.addEventListener('change', async () => {
            await fetchPredictStations(els.predictModelSelect.value);
            populatePredictionControls();
            checkPredictionCapability();
        });

        // Mode picker (Historical Fit / Future Forecast)
        if (els.predictModePicker) {
            els.predictModePicker.querySelectorAll('.mode-seg-btn').forEach(btn => {
                btn.addEventListener('click', () => {
                    state.predictMode = btn.dataset.mode;
                    els.predictModePicker.querySelectorAll('.mode-seg-btn').forEach(b => b.classList.remove('active'));
                    btn.classList.add('active');
                    if (state.predictMode === 'historical') {
                        els.predictHorizonLabel.textContent = 'Horizon H (1–30)';
                        els.predictHorizonInput.removeAttribute('max');
                        validateHorizonInput();
                    } else {
                        els.predictHorizonLabel.textContent = 'Horizon (1–30)';
                        els.predictHorizonInput.removeAttribute('max');
                        validateHorizonInput();
                    }
                    populatePredictionControls();
                    checkPredictionCapability();
                });
            });
        }

        // Horizon input hint
        function validateHorizonInput() {
            let val = parseInt(els.predictHorizonInput.value, 10);
            if (isNaN(val) || val < 1 || val > 30) {
                els.predictionMessage.textContent = 'Horizon must be between 1 and 30 steps.';
                els.predictionMessage.className = 'inline-message warn';
            } else {
                els.predictionMessage.textContent = '';
                els.predictionMessage.className = 'inline-message';
            }
        }
        els.predictHorizonInput.addEventListener('input', validateHorizonInput);
        els.predictHorizonInput.addEventListener('change', validateHorizonInput);

        // CI band toggle — show/hide trace index 2 on all prediction forecast plots
        els.predictCIToggle?.addEventListener('change', () => {
            const showCI = els.predictCIToggle.checked;
            els.predictionCards.querySelectorAll('.plot-container[data-has-ci="true"]').forEach(div => {
                if (div.data && div.data.length > 2) Plotly.restyle(div, { visible: showCI }, [2]);
            });
        });

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

        // Initialize help icon tooltips
        initHelpTooltips();
    }

    function initHelpTooltips() {
        let activeTooltip = null;

        document.querySelectorAll('.help-icon').forEach(icon => {
            icon.addEventListener('mouseenter', () => {
                // Remove previous tooltip if exists
                if (activeTooltip) activeTooltip.remove();

                const tooltip = document.createElement('div');
                tooltip.className = 'tooltip-popover';
                tooltip.textContent = icon.getAttribute('title');
                document.body.appendChild(tooltip);

                // Position tooltip above the icon
                const rect = icon.getBoundingClientRect();
                const tooltipRect = tooltip.getBoundingClientRect();
                const top = rect.top - tooltipRect.height - 8;
                const left = rect.left + rect.width / 2 - tooltipRect.width / 2;

                tooltip.style.top = Math.max(8, top) + 'px';
                tooltip.style.left = Math.max(8, Math.min(left, window.innerWidth - tooltipRect.width - 8)) + 'px';

                activeTooltip = tooltip;
            });

            icon.addEventListener('mouseleave', () => {
                if (activeTooltip) {
                    activeTooltip.remove();
                    activeTooltip = null;
                }
            });
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
        const res = await fetch('/api/datasets');
        const folderNames = await res.json();

        const keyMap = { lamah: 'lamah', mekong: 'mekong' };
        const datasets = folderNames.map(name => ({
            label: name,
            key: keyMap[name.toLowerCase()] ?? name.toLowerCase(),
        }));
        const allDatasets = [{ label: 'All', key: 'all' }, ...datasets];

        state.activeDatasetFilter = 'all';

        const containers = [els.datasetPicker, els.freeDatasetPicker].filter(Boolean);

        function syncActiveStyling(selectedKey) {
            containers.forEach(container => {
                Array.from(container.children).forEach(c =>
                    c.classList.toggle('active', c.dataset.dsKey === selectedKey));
            });
        }

        containers.forEach(container => {
            container.innerHTML = '';
            allDatasets.forEach(({ label, key }) => {
                const btn = document.createElement('button');
                btn.className = `feature-chip${key === state.activeDatasetFilter ? ' active' : ''}`;
                btn.dataset.dsKey = key;
                btn.textContent = label;
                btn.addEventListener('click', () => {
                    if (state.activeDatasetFilter === key) return;
                    state.activeDatasetFilter = key;
                    syncActiveStyling(key);
                    // Sync the map filter bar too
                    Array.from(els.datasetFilterBar?.children ?? []).forEach(c => {
                        c.classList.toggle('active', c.textContent.toLowerCase().includes(key) || (key === 'all' && c.textContent.toLowerCase() === 'all datasets'));
                    });
                    buildFeatureFilterBar();
                    refreshMarkerVisibility();
                    // Refresh station dropdowns in charted builder and re-sync features
                    els.seriesBuilder.querySelectorAll('.series-row').forEach(row => {
                        const stationSel = row.querySelector('.series-station');
                        if (!stationSel) return;
                        fillStationSelect(stationSel, stationSel.value);
                        // Re-sync features to match the (possibly new) selected station
                        syncRowFeatureOptions(row);
                        syncRowDateBounds(row);
                    });
                    refreshDisabledOptions(els.seriesBuilder, '.series-station', '.series-feature');
                    // Also refresh feature-first station selects
                    const ffContainer = els.seriesBuilder.querySelector('.feature-first-container');
                    if (ffContainer) {
                        const ffFeature = ffContainer.querySelector('.ff-feature-select')?.value;
                        ffContainer.querySelectorAll('.sr-station-select').forEach(sel => {
                            ffPopulateStationSelect(sel, ffFeature);
                        });
                    }
                    // Reset free-form builder: clear all rows and add one fresh row
                    els.freeAnalysisBuilder.innerHTML = '';
                    addFreeSeriesRow();
                    // Fly map to the chosen dataset
                    if (key === 'mekong' && state.geojsonLayer) {
                        const b = state.geojsonLayer.getBounds();
                        if (b.isValid()) state.map.flyToBounds(b.pad(0.15), { duration: 1.2 });
                    } else if (key === 'lamah' && state.lamahGeojsonLayer) {
                        const b = state.lamahGeojsonLayer.getBounds();
                        if (b.isValid()) state.map.flyToBounds(b.pad(0.1), { duration: 1.2 });
                    }
                });
                container.appendChild(btn);
            });
        });

        // Compare panel has its own independent dataset picker (no 'all')
        if (els.compareDatasetPicker) {
            state.compareDataset = datasets[0]?.key ?? 'mekong';
            els.compareDatasetPicker.innerHTML = '';
            datasets.forEach(({ label, key }) => {
                const btn = document.createElement('button');
                btn.className = `feature-chip${key === state.compareDataset ? ' active' : ''}`;
                btn.dataset.dsKey = key;
                btn.textContent = label;
                btn.addEventListener('click', () => {
                    if (state.compareDataset === key) return;
                    state.compareDataset = key;
                    Array.from(els.compareDatasetPicker.children).forEach(c =>
                        c.classList.toggle('active', c.dataset.dsKey === key));
                    populateCompareFeatures();
                });
                els.compareDatasetPicker.appendChild(btn);
            });
            populateCompareFeatures();
        }

        els.compareFeatureSelect?.addEventListener('change', updateCompareYearHint);

        // Prediction panel has its own independent dataset picker
        if (els.predictDatasetPicker) {
            els.predictDatasetPicker.innerHTML = '';
            allDatasets.forEach(({ label, key }) => {
                const btn = document.createElement('button');
                btn.className = `feature-chip${key === state.predictDatasetFilter ? ' active' : ''}`;
                btn.dataset.dsKey = key;
                btn.textContent = label;
                btn.addEventListener('click', () => {
                    if (state.predictDatasetFilter === key) return;
                    state.predictDatasetFilter = key;
                    Array.from(els.predictDatasetPicker.children).forEach(c =>
                        c.classList.toggle('active', c.dataset.dsKey === key));
                    populatePredictionControls();
                });
                els.predictDatasetPicker.appendChild(btn);
            });
        }
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

    async function populateModelDropdown() {
        try {
            const res = await fetch('/api/predict-models');
            const models = await res.json();
            els.predictModelSelect.innerHTML = '';
            if (!models || models.length === 0) {
                const opt = document.createElement('option');
                opt.value = '';
                opt.disabled = true;
                opt.selected = true;
                opt.textContent = 'No trained models available';
                els.predictModelSelect.appendChild(opt);
                return;
            }
            models.forEach(name => {
                const opt = document.createElement('option');
                opt.value = name;
                opt.textContent = name;
                if (name === 'FlowNet') opt.selected = true;
                els.predictModelSelect.appendChild(opt);
            });
        } catch (e) {
            // fallback: single default
            els.predictModelSelect.innerHTML = '<option value="FlowNet" selected>FlowNet</option>';
        }
    }

    async function fetchPredictStations(model) {
        try {
            const res = await fetch(`/api/predict-stations?model=${encodeURIComponent(model)}`);
            const data = await res.json();
            state.predictStationsForModel = {
                lamah: {
                    historical: new Set((data.lamah?.historical) || []),
                    future: new Set((data.lamah?.future) || []),
                },
                mekong: {
                    historical: new Set((data.mekong?.historical) || []),
                    future: new Set((data.mekong?.future) || []),
                },
            };
        } catch (e) {
            state.predictStationsForModel = {
                lamah: { historical: new Set(), future: new Set() },
                mekong: { historical: new Set(), future: new Set() },
            };
        }
    }

    function populatePredictionControls() {
        const ds = state.predictDatasetFilter;
        const prev = els.predictStationSelect.value;
        const mode = state.predictMode;
        const { lamah, mekong } = state.predictStationsForModel;
        els.predictStationSelect.innerHTML = '';
        state.bootstrap.station_names.forEach(stationName => {
            const meta = state.stationsByName.get(stationName);
            if (ds !== 'all' && (!meta || meta.dataset !== ds)) return;
            // Only include stations that have prediction files for the selected model + mode
            const dataset = meta?.dataset;
            if (dataset === 'lamah' && !lamah[mode]?.has(stationName)) return;
            if (dataset === 'mekong' && !mekong[mode]?.has(stationName)) return;
            const option = document.createElement('option');
            option.value = stationName;
            option.textContent = prettyStation(stationName);
            if (stationName === prev) option.selected = true;
            els.predictStationSelect.appendChild(option);
        });
        updatePredictionFeatureOptions();
        checkPredictionCapability();
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
        checkPredictionCapability();
    }

    /**
     * Check if the current station/feature/mode/model combo has a trained CSV.
     * Shows or hides a warning banner in the prediction controls panel.
     */
    function checkPredictionCapability() {
        const caps = state.bootstrap?.prediction_capabilities;
        if (!caps) return;

        const station = els.predictStationSelect.value;
        const feature = els.predictFeatureSelect.value;
        const mode = state.predictMode;
        const meta = state.stationsByName.get(station);
        const dataset = meta?.dataset;
        if (!dataset || !feature) return;

        let modelsTrained = [];
        if (dataset === 'mekong') {
            const mekongFeatureFolder = {
                'Discharge': 'Water_Discharge',
                'Water_Level': 'Water_Level',
                'Rainfall': 'Rainfall',
                'Total_Suspended_Solids': 'Total_Suspended_Solids',
            };
            const featureKey = mekongFeatureFolder[feature] || feature;
            modelsTrained = caps.mekong?.[mode]?.[featureKey] || [];
        } else if (dataset === 'lamah') {
            modelsTrained = caps.lamah?.[mode] || [];
        }

        // Find or create the capability warning element
        let warnEl = document.getElementById('predict-capability-warning');
        if (!warnEl) {
            warnEl = document.createElement('div');
            warnEl.id = 'predict-capability-warning';
            warnEl.className = 'capability-warning-banner';
            // Insert just above the Run button
            const runBtn = els.runPredictionBtn;
            if (runBtn?.parentNode) runBtn.parentNode.insertBefore(warnEl, runBtn);
        }

        if (modelsTrained.length === 0) {
            warnEl.innerHTML = `⚠️ <strong>No trained model artifacts</strong> for <em>${escapeHtml(prettyFeature(feature))}</em> on ${escapeHtml(dataset.charAt(0).toUpperCase() + dataset.slice(1))} in ${mode} mode. The forecast will use a <strong>Holt-Winters statistical projection</strong> instead of a trained ML model.`;
            warnEl.style.display = 'block';
        } else {
            warnEl.style.display = 'none';
        }
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
        const metaLabel = fragment.querySelector('.series-row-meta');
        const removeBtn = row.querySelector('.series-remove-btn');

        // Smart default when no prefill: pick next unused station/feature
        const nextUnused = prefill ? null : pickNextUnused(els.seriesBuilder, '.series-station', '.series-feature');
        fillStationSelect(stationSelect, prefill?.station ?? nextUnused?.station);
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
            refreshDisabledOptions(els.seriesBuilder, '.series-station', '.series-feature');
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
            refreshDisabledOptions(els.seriesBuilder, '.series-station', '.series-feature');
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
                refreshDisabledOptions(els.seriesBuilder, '.series-station', '.series-feature');
            });
        }

        els.seriesBuilder.appendChild(fragment);
        const insertedRow = els.seriesBuilder.lastElementChild;
        if (prefill?.feature) {
            syncRowFeatureOptions(insertedRow, prefill.feature);
        } else {
            syncRowFeatureOptions(insertedRow, nextUnused?.feature);
        }
        if (prefill?.start_date) insertedRow.querySelector('.series-start').value = prefill.start_date;
        if (prefill?.end_date) insertedRow.querySelector('.series-end').value = prefill.end_date;
        syncRowDateBounds(insertedRow);
        attachDateEnforcement(insertedRow, '.series-start', '.series-end');
        metaLabel.textContent = 'The row metadata updates automatically based on the station and feature you choose.';
        refreshDisabledOptions(els.seriesBuilder, '.series-station', '.series-feature');
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

    // ── Series builder shared utilities ────────────────────────────────────

    /** Returns a set of "station|feature" strings currently in use across all rows. */
    function getUsedPairs(builderEl, stationClass, featureClass) {
        const used = new Set();
        builderEl.querySelectorAll('.series-row').forEach(row => {
            const s = row.querySelector(stationClass)?.value;
            const f = row.querySelector(featureClass)?.value;
            if (s && f) used.add(`${s}|${f}`);
        });
        return used;
    }

    /**
     * Pick the next station/feature combo not yet used in the builder.
     * Prefers adding a new feature to the last row's station; falls back to another station.
     */
    function pickNextUnused(builderEl, stationClass, featureClass) {
        const used = getUsedPairs(builderEl, stationClass, featureClass);
        const ds = state.activeDatasetFilter;

        // Prefer last row's station (only if it belongs to the active dataset)
        const rows = builderEl.querySelectorAll('.series-row');
        const lastStation = rows.length ? rows[rows.length - 1].querySelector(stationClass)?.value : null;
        if (lastStation) {
            const lastMeta = state.stationsByName.get(lastStation);
            const inDataset = ds === 'all' || lastMeta?.dataset === ds;
            if (inDataset) {
                for (const f of (lastMeta?.features || [])) {
                    if (!used.has(`${lastStation}|${f}`)) return { station: lastStation, feature: f };
                }
            }
        }

        // Fall back to any station with an unused feature
        for (const stName of (state.bootstrap?.station_names || [])) {
            if (ds !== 'all') {
                const m = state.stationsByName.get(stName);
                if (!m || m.dataset !== ds) continue;
            }
            const meta = state.stationsByName.get(stName);
            for (const f of (meta?.features || [])) {
                if (!used.has(`${stName}|${f}`)) return { station: stName, feature: f };
            }
        }
        return null;
    }

    /**
     * Disable feature options already in use by other rows for the same station,
     * and disable station options where every feature is already taken by other rows.
     */
    function refreshDisabledOptions(builderEl, stationClass, featureClass) {
        const allRows = Array.from(builderEl.querySelectorAll('.series-row'));
        allRows.forEach(row => {
            const myStation = row.querySelector(stationClass)?.value;
            const myFeature = row.querySelector(featureClass)?.value;
            const featureSel = row.querySelector(featureClass);
            const stationSel = row.querySelector(stationClass);

            // Map: station -> Set<feature> used by OTHER rows
            const usedByOthers = {};
            allRows.forEach(r => {
                if (r === row) return;
                const s = r.querySelector(stationClass)?.value;
                const f = r.querySelector(featureClass)?.value;
                if (s && f) {
                    if (!usedByOthers[s]) usedByOthers[s] = new Set();
                    usedByOthers[s].add(f);
                }
            });

            if (featureSel) {
                Array.from(featureSel.options).forEach(opt => {
                    opt.disabled = opt.value !== myFeature && !!(usedByOthers[myStation]?.has(opt.value));
                });
            }
            if (stationSel) {
                Array.from(stationSel.options).forEach(opt => {
                    const meta = state.stationsByName.get(opt.value);
                    const allFeatures = meta?.features || [];
                    const taken = usedByOthers[opt.value] || new Set();
                    // Disable only if EVERY feature is taken — unless this row currently uses that station
                    opt.disabled = opt.value !== myStation &&
                        allFeatures.length > 0 &&
                        allFeatures.every(f => taken.has(f));
                });
            }
        });
    }

    // ── Free-form analysis builder ─────────────────────────────────────────
    function addFreeSeriesRow() {
        const row = document.createElement('div');
        row.className = 'series-row';
        row.innerHTML = `
            <div class="series-row-top">
                <strong class="series-row-label">Series</strong>
                <button class="ghost-btn icon-only free-row-remove-btn" type="button" aria-label="Remove" style="color:#ef4444;">✕</button>
            </div>
            <div class="control-grid four-col">
                <div class="control-group station-group">
                    <label>Station</label>
                    <select class="free-station"></select>
                </div>
                <div class="control-group feature-group">
                    <label>Feature</label>
                    <select class="free-feature"></select>
                </div>
                <div class="control-group shared-dates">
                    <label>Start date</label>
                    <input class="free-start" type="date" />
                </div>
                <div class="control-group shared-dates">
                    <label>End date</label>
                    <input class="free-end" type="date" />
                </div>
            </div>
            <div class="series-row-meta free-row-meta"></div>`;

        const stationSel = row.querySelector('.free-station');
        const featureSel = row.querySelector('.free-feature');
        const startInput = row.querySelector('.free-start');
        const endInput   = row.querySelector('.free-end');
        const metaEl     = row.querySelector('.free-row-meta');
        const removeBtn  = row.querySelector('.free-row-remove-btn');

        // Smart default: pick next unused station/feature combo
        const next = pickNextUnused(els.freeAnalysisBuilder, '.free-station', '.free-feature');
        fillStationSelect(stationSel, next?.station);
        syncFreeFeatureOptions(stationSel, featureSel, startInput, endInput, metaEl);
        if (next?.feature && Array.from(featureSel.options).some(o => o.value === next.feature)) {
            featureSel.value = next.feature;
            syncFreeDateBounds(stationSel, featureSel, startInput, endInput, metaEl);
        }

        stationSel.addEventListener('change', () => {
            syncFreeFeatureOptions(stationSel, featureSel, startInput, endInput, metaEl);
            refreshDisabledOptions(els.freeAnalysisBuilder, '.free-station', '.free-feature');
        });
        featureSel.addEventListener('change', () => {
            syncFreeDateBounds(stationSel, featureSel, startInput, endInput, metaEl);
            refreshDisabledOptions(els.freeAnalysisBuilder, '.free-station', '.free-feature');
        });
        removeBtn.addEventListener('click', () => {
            row.remove();
            updateFreeRemoveBtns();
            refreshDisabledOptions(els.freeAnalysisBuilder, '.free-station', '.free-feature');
        });

        els.freeAnalysisBuilder.appendChild(row);
        attachDateEnforcement(row, '.free-start', '.free-end');
        updateFreeRemoveBtns();
        refreshDisabledOptions(els.freeAnalysisBuilder, '.free-station', '.free-feature');
    }

    function syncFreeFeatureOptions(stationSel, featureSel, startInput, endInput, metaEl) {
        const meta = state.stationsByName.get(stationSel.value);
        const features = meta?.features || [];
        const prev = featureSel.value;
        featureSel.innerHTML = '';
        features.forEach(f => {
            const opt = document.createElement('option');
            opt.value = f;
            opt.textContent = prettyFeature(f);
            if (f === prev) opt.selected = true;
            featureSel.appendChild(opt);
        });
        syncFreeDateBounds(stationSel, featureSel, startInput, endInput, metaEl);
    }

    function syncFreeDateBounds(stationSel, featureSel, startInput, endInput, metaEl) {
        const meta = state.stationsByName.get(stationSel.value);
        const feature = featureSel.value;
        if (!meta || !feature) return;
        const detail = meta.feature_details?.[feature];
        if (!detail) return;
        const cappedEnd = capDate(detail.end_date);
        startInput.min = detail.start_date;
        startInput.max = cappedEnd;
        endInput.min   = detail.start_date;
        endInput.max   = cappedEnd;
        // Always reset to full dataset range when station or feature changes
        startInput.value = detail.start_date;
        endInput.value   = cappedEnd;
        metaEl.textContent = `Data range: ${detail.start_date} → ${cappedEnd}`;
    }

    function updateFreeRemoveBtns() {
        const rows = els.freeAnalysisBuilder.querySelectorAll('.series-row');
        rows.forEach(r =>
            r.querySelector('.free-row-remove-btn').classList.toggle('hidden', rows.length === 1));
    }


    async function submitFreeAnalysis() {
        const rows = Array.from(els.freeAnalysisBuilder.querySelectorAll('.series-row'));
        const series = rows.map(row => ({
            station:    row.querySelector('.free-station').value,
            feature:    row.querySelector('.free-feature').value,
            start_date: row.querySelector('.free-start').value,
            end_date:   row.querySelector('.free-end').value,
        }));
        if (series.some(s => !s.station || !s.feature || !s.start_date || !s.end_date)) {
            showMessage(els.freeAnalysisMessage, 'Please fill in all fields for each series.', 'error');
            return;
        }
        const btn = els.runFreeAnalysisBtn;
        const originalText = btn.textContent;
        const includeAnalysis = document.getElementById('analysisAnalysisToggle')?.checked ?? false;
        btn.disabled = true;
        btn.textContent = 'Analysing…';
        showMessage(els.freeAnalysisMessage, 'Processing analysis… this may take a moment.', '');
        try {
            const response = await fetch('/api/analyze-free-multi', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ series, include_analysis: includeAnalysis }),
            });
            const data = await response.json();
            if (!response.ok || !data.ok) throw new Error(data.error || 'Analysis failed.');
            appendFreeMultiAnalysisCard(data.result.graphs, data.result.benchmark, data.result.benchmark_analysis, data.result.analysis);
            activateDockTab('analysis');
            showMessage(els.freeAnalysisMessage, `${data.result.graphs.length} charts generated — see Analysis panel.`, 'success');
        } catch (err) {
            showMessage(els.freeAnalysisMessage, err.message || 'Something went wrong.', 'error');
        } finally {
            btn.disabled = false;
            btn.textContent = originalText;
        }
    }

    function appendFreeMultiAnalysisCard(graphs, benchmark, benchmarkAnalysis, extraAnalysis) {
        if (!graphs || !graphs.length) return;
        clearEmptyStateIfNeeded(els.analysisCards);

        const cardId = `analysis-${++state.cardCounters.analysis}`;

        // Build the outer story card
        const card = document.createElement('article');
        card.className = 'workspace-card';
        card.innerHTML = `
            <div class="workspace-card-header">
                <div style="flex:1;min-width:0;">
                    <h3 class="workspace-card-title">📊 Analysis Report</h3>
                    <div class="workspace-card-subtitle">${escapeHtml(describeSeries(graphs[0].series))} · ${graphs.length} complementary views</div>
                </div>
                <div class="card-header-actions" style="display:flex;gap:8px;flex-shrink:0;align-items:start;">
                    <button class="expand-btn ghost-btn icon-only" type="button" title="Expand to fullscreen">
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="15 3 21 3 21 9"/><polyline points="9 21 3 21 3 15"/><line x1="21" y1="3" x2="14" y2="10"/><line x1="3" y1="21" x2="10" y2="14"/></svg>
                    </button>
                    <button class="delete-btn ghost-btn" type="button">Delete</button>
                </div>
            </div>
            <div class="workspace-card-body report-body" id="${cardId}"></div>
        `;
        card.querySelector('.expand-btn')?.addEventListener('click', () => openFullscreen(card));
        card.querySelector('.delete-btn')?.addEventListener('click', () => {
            const parent = card.parentElement;
            card.remove();
            if (!parent.children.length) {
                setEmptyState(els.analysisCards, 'No analysis cards yet. Build a selection and choose "Add to analyze".');
            }
        });

        const body = card.querySelector('.report-body');

        if (benchmark && benchmark.length) {
            const benchmarkEl = document.createElement('div');
            benchmarkEl.innerHTML = buildBenchmarkTables(benchmark, benchmarkAnalysis);
            body.appendChild(benchmarkEl);
        }

        graphs.forEach((graph, idx) => {
            const label = graph.graph_label || graph.graph_type;
            const isLast = idx === graphs.length - 1;

            const section = document.createElement('div');
            section.className = 'report-section' + (isLast ? ' report-section-last' : '');
            section.innerHTML = `
                <div class="report-section-heading">
                    <span class="report-step">${idx + 1}</span>
                    <span class="report-section-title">${escapeHtml(label)}</span>
                </div>
                <div class="plot-container" id="${cardId}-g${idx}"></div>
                <div class="report-analysis"></div>
            `;
            body.appendChild(section);
            renderPlot(section.querySelector('.plot-container'), graph.figure);
            section.querySelector('.report-analysis').innerHTML = graph.analysis?.summary || '';
        });

        els.analysisCards.prepend(card);
        refreshPlotGrid(els.analysisCards);
        if (extraAnalysis) appendAnalysisSection(card.querySelector('.workspace-card-body'), extraAnalysis);
    }
    // ── End free-form analysis builder ────────────────────────────────────

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

        const ds = state.activeDatasetFilter;
        state.bootstrap.station_names.forEach((stationName) => {
            const meta = state.stationsByName.get(stationName);
            if (ds !== 'all' && (!meta || meta.dataset !== ds)) return;
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
            const cappedEnd = capDate(detail.end_date);
            startInput.min = detail.start_date;
            startInput.max = cappedEnd;
            endInput.min = detail.start_date;
            endInput.max = cappedEnd;
            if (!startInput.value || startInput.value < detail.start_date || startInput.value > cappedEnd) {
                startInput.value = detail.start_date;
            }
            if (!endInput.value || endInput.value > cappedEnd || endInput.value < detail.start_date) {
                endInput.value = cappedEnd;
            }
            if (startInput.value > endInput.value) {
                startInput.value = detail.start_date;
                endInput.value = cappedEnd;
            }

            if (stationSelect.multiple && stationSelect.selectedOptions.length > 1) {
                metaEl.textContent = `Coverage approx based on first station: ${detail.start_date} → ${detail.end_date}`;
            } else {
                metaEl.textContent = `Coverage: ${detail.start_date} → ${detail.end_date} · ${detail.observations.toLocaleString()} observations · ${detail.imputed_points.toLocaleString()} imputed · Unit: ${detail.unit}`;
            }
        }
        // Fallback for regular dates if somehow out of bounds
        if (!isAnnualOverview) {
            const cappedEnd2 = capDate(detail.end_date);
            if (forceReset) {
                startInput.value = detail.start_date;
                endInput.value = cappedEnd2;
            } else {
                if (!startInput.value || startInput.value < detail.start_date || startInput.value > cappedEnd2) {
                    startInput.value = detail.start_date;
                }
                if (!endInput.value || endInput.value > cappedEnd2 || endInput.value < detail.start_date) {
                    endInput.value = cappedEnd2;
                }
                if (startInput.value > endInput.value) {
                    startInput.value = detail.start_date;
                    endInput.value = cappedEnd2;
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

            const cappedEnd = capDate(detail.end_date);
            div.innerHTML = `
                <div class="control-group" style="margin-bottom: 8px;">
                    <label style="color: var(--primary);">Dates for ${prettyFeature(feature)}</label>
                    <div style="font-size: 11px; color: var(--text-muted); margin-top: 2px;">Available: ${detail.start_date} → ${detail.end_date}</div>
                </div>
                <div class="control-grid two-col">
                    <div class="control-group">
                        <label>Start date</label>
                        <input class="pf-start" type="date" value="${detail.start_date}" min="${detail.start_date}" max="${cappedEnd}" />
                    </div>
                    <div class="control-group">
                        <label>End date</label>
                        <input class="pf-end" type="date" value="${cappedEnd}" min="${detail.start_date}" max="${cappedEnd}" />
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
            // Get all available stations for this feature (respecting dataset filter)
            const ds = state.activeDatasetFilter;
            const availableStations = state.bootstrap.station_names.filter(name => {
                const meta = state.stationsByName.get(name);
                if (!meta || !meta.features.includes(feature)) return false;
                return ds === 'all' || meta.dataset === ds;
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
        const cappedEnd = capDate(detail.end_date);
        startInput.min = detail.start_date;
        startInput.max = cappedEnd;
        endInput.min = detail.start_date;
        endInput.max = cappedEnd;
        if (!startInput.value || startInput.value < detail.start_date || startInput.value > cappedEnd) {
            startInput.value = detail.start_date;
        }
        if (!endInput.value || endInput.value > cappedEnd || endInput.value < detail.start_date) {
            endInput.value = cappedEnd;
        }
        if (startInput.value > endInput.value) {
            startInput.value = detail.start_date;
            endInput.value = cappedEnd;
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
            console.error('[submitSeriesRequest] error:', error);
            showMessage(targetMessage, error.message || 'Something went wrong.', 'error');
        }
    }

    async function runPrediction() {
        const station = els.predictStationSelect.value;
        const feature = els.predictFeatureSelect.value;
        const horizonRaw = els.predictHorizonInput.value?.trim();
        const horizon = horizonRaw ? Number(horizonRaw) : 0;

        // Validate horizon is within range
        if (isNaN(horizon) || horizon < 1 || horizon > 30) {
            showMessage(els.predictionMessage, 'Horizon must be between 1 and 30 steps.', 'error');
            return;
        }

        const payload = {
            station,
            feature,
            horizon,
            model: els.predictModelSelect.value,
            mode: state.predictMode,
            analysis: els.predictAnalysisToggle?.checked ?? true,
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
        const plot = card.querySelector('.plot-container');
        plot.dataset.graphType = result.graph_type || '';
        if (result.graph_type === 'Calendar Heatmap') {
            plot.style.minHeight = '620px';
        }
        addCsvButton(card, result.series);
        els.visualizationCards.prepend(card);
        renderPlot(plot, result.figure);
        requestAnimationFrame(() => {
            els.visualizationCards.querySelectorAll('.plot-container[data-figure]').forEach((existingPlot) => {
                renderPlot(existingPlot, existingPlot.dataset.figure);
            });
        });
    }

    function buildBenchmarkTables(benchmark, benchmarkAnalysis) {
        if (!benchmark || !benchmark.length) return '';

        // Group by dataset
        const byDataset = {};
        benchmark.forEach(b => {
            const key = b.dataset;
            if (!byDataset[key]) byDataset[key] = [];
            byDataset[key].push(b);
        });

        let html = '<div class="benchmark-section">';
        html += '<h4 class="benchmark-heading">Dataset Benchmark</h4>';

        for (const [dataset, rows] of Object.entries(byDataset)) {
            const n = rows[0].n_stations;
            html += `<div class="benchmark-dataset-label">${escapeHtml(dataset.charAt(0).toUpperCase() + dataset.slice(1))} dataset — ${n} stations</div>`;
            html += '<div class="benchmark-table-wrap"><table class="benchmark-table">';
            html += '<thead><tr><th>Station</th><th>Feature</th><th>Station Mean</th><th>Dataset Mean</th><th>Difference</th><th>Z-score</th><th>Percentile Rank</th></tr></thead>';
            html += '<tbody>';
            rows.forEach(b => {
                const diff = b.pct_diff;
                const diffClass = diff > 0 ? 'bench-above' : diff < 0 ? 'bench-below' : '';
                const diffText = diff > 0 ? `+${diff.toFixed(1)}%` : `${diff.toFixed(1)}%`;
                const zText = b.z_score > 0 ? `+${b.z_score.toFixed(2)}` : b.z_score.toFixed(2);
                const rank = b.percentile_rank;
                const rankSuffix = rank === 1 ? 'st' : rank === 2 ? 'nd' : rank === 3 ? 'rd' : 'th';
                html += `<tr>
                    <td>${escapeHtml(b.station_label)}</td>
                    <td>${escapeHtml(b.feature_label)}</td>
                    <td>${b.station_mean} ${escapeHtml(b.unit)}</td>
                    <td>${b.dataset_mean} ${escapeHtml(b.unit)}</td>
                    <td class="${diffClass}">${diffText}</td>
                    <td class="${diffClass}">${zText}</td>
                    <td>${rank}${rankSuffix}</td>
                </tr>`;
            });
            html += '</tbody></table></div>';
        }
        if (benchmarkAnalysis) {
            html += `<div class="benchmark-analysis">${benchmarkAnalysis}</div>`;
        }
        html += '</div>';
        return html;
    }

    function appendAnalysisCard(result) {
        clearEmptyStateIfNeeded(els.analysisCards);
        const cardId = `analysis-${++state.cardCounters.analysis}`;
        const card = buildBaseCard(cardId, result.title, describeSeries(result.series));
        addCsvButton(card, result.series);
        const body = card.querySelector('.workspace-card-body');
        const analysis = result.analysis;
        const analysisBlock = document.createElement('div');
        analysisBlock.className = 'analysis-block';
        analysisBlock.innerHTML = `
            <h4>📊 Summary</h4>
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
        const hasAiSummary = analysis.summary && (analysis.summary.includes('<p>') || analysis.summary.includes('<ul>') || analysis.summary.includes('<li>'));
        if (!hasAiSummary && analysis.comparisons && analysis.comparisons.length) {
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
        if (analysis.benchmark && analysis.benchmark.length) {
            const benchmarkEl = document.createElement('div');
            benchmarkEl.innerHTML = buildBenchmarkTables(analysis.benchmark, analysis.benchmark_analysis);
            analysisBlock.appendChild(benchmarkEl);
        }
        refreshPlotGrid(els.analysisCards);
    }

    function appendPredictionCard(result) {
        clearEmptyStateIfNeeded(els.predictionCards);
        const horizonLabel = `${result.horizon} ${result.frequency === 'monthly' ? 'month(s)' : 'day(s)'}`;
        const subtitle = `${prettyStation(result.station)} · ${prettyFeature(result.feature)} · Horizon ${horizonLabel}`;
        const cardId = `prediction-${++state.cardCounters.prediction}`;
        const card = buildBaseCard(cardId, result.title, subtitle);
        const body = card.querySelector('.workspace-card-body');
        // basePlot is the plot-container created by buildBaseCard (used for zoom — shown in fullscreen)
        const basePlot = body.querySelector('.plot-container');

        // ── Metrics badge — inserted at the very top of the card body ──────────
        if (result.model_metrics) {
            const m = result.model_metrics;
            const metricsEl = document.createElement('div');
            metricsEl.className = 'pred-metrics-strip';
            const rmseVal = m.rmse != null ? m.rmse : '—';
            const mapeVal = m.mape != null ? `${m.mape}%` : '—';
            metricsEl.innerHTML =
                `<span class="pms-item"><span class="pms-label">Horizon</span><span class="pms-value">${escapeHtml(horizonLabel)}</span></span>` +
                `<span class="pms-divider"></span>` +
                `<span class="pms-item"><span class="pms-label">RMSE</span><span class="pms-value">${escapeHtml(String(rmseVal))}</span></span>` +
                `<span class="pms-divider"></span>` +
                `<span class="pms-item"><span class="pms-label">MAPE</span><span class="pms-value">${escapeHtml(String(mapeVal))}</span></span>`;
            body.insertBefore(metricsEl, basePlot);
        }

        // Insert zoom label BEFORE basePlot
        const zoomLabel = document.createElement('div');
        zoomLabel.className = 'chart-section-label';
        zoomLabel.textContent = 'Zoomed view · Last 3 months + forecast';
        body.insertBefore(zoomLabel, basePlot);

        // Mark basePlot as having a CI band (trace index 2)
        basePlot.dataset.hasCi = 'true';

        // Full history section — appended after basePlot
        const fullLabel = document.createElement('div');
        fullLabel.className = 'chart-section-label';
        fullLabel.textContent = 'Full historical view';
        body.appendChild(fullLabel);

        const fullPlot = document.createElement('div');
        fullPlot.className = 'plot-container';
        fullPlot.dataset.hasCi = 'true';
        body.appendChild(fullPlot);

        // Analysis block
        const block = document.createElement('div');
        block.className = 'analysis-block';
        block.innerHTML = `<h4>Prediction Analysis</h4><div class="analysis-summary"></div>`;
        const summaryEl = block.querySelector('.analysis-summary');
        if (result.summary) {
            if (result.summary.includes('<p>') || result.summary.includes('<ul>') || result.summary.includes('<li>')) {
                summaryEl.innerHTML = result.summary.replace(/^🧠 \*\*AI Analysis\*\*\n\n/, '').replace(/^🧠 Analysis:\n/, '').replace(/^Analysis:\n/, '');
            } else {
                summaryEl.textContent = result.summary;
            }
            body.appendChild(block);
        }

        // Residual diagnostics panel
        if (result.figure_diagnostics) {
            const diagLabel = document.createElement('div');
            diagLabel.className = 'chart-section-label';
            diagLabel.textContent = 'Residual diagnostics · ACF · PACF';
            body.appendChild(diagLabel);
            const diagPlot = document.createElement('div');
            diagPlot.className = 'plot-container';
            body.appendChild(diagPlot);
            if (result.diagnostics_summary) {
                const diagBlock = document.createElement('div');
                diagBlock.className = 'analysis-block';
                diagBlock.innerHTML = `<h4>Diagnostics Interpretation</h4><div class="analysis-summary"></div>`;
                const diagSummaryEl = diagBlock.querySelector('.analysis-summary');
                if (result.diagnostics_summary.includes('<p>') || result.diagnostics_summary.includes('<ul>') || result.diagnostics_summary.includes('<li>')) {
                    diagSummaryEl.innerHTML = result.diagnostics_summary;
                } else {
                    diagSummaryEl.textContent = result.diagnostics_summary;
                }
                body.appendChild(diagBlock);
            }
        }

        // Prepend to DOM first so Plotly can measure container width correctly
        els.predictionCards.prepend(card);

        // Render zoom into basePlot (first .plot-container → used by openFullscreen)
        renderPlot(basePlot, result.figure_zoom || result.figure);
        // Render full history into the separate fullPlot div
        renderPlot(fullPlot, result.figure);
        // Render diagnostics
        if (result.figure_diagnostics) {
            const diagPlot = body.querySelectorAll('.plot-container')[2];
            if (diagPlot) renderPlot(diagPlot, result.figure_diagnostics);
        }
        refreshPlotGrid(els.predictionCards);
    }

    function buildBaseCard(cardId, title, subtitle) {
        const card = document.createElement('article');
        card.className = 'workspace-card';

        // ── Header ────────────────────────────────────────────────────────────
        const header = document.createElement('div');
        header.className = 'workspace-card-header';

        // Title block (left)
        const titleBlock = document.createElement('div');
        titleBlock.className = 'workspace-card-title-block';

        const titleEl = document.createElement('h3');
        titleEl.className = 'workspace-card-title';
        titleEl.textContent = title;

        const subtitleEl = document.createElement('div');
        subtitleEl.className = 'workspace-card-subtitle';
        subtitleEl.textContent = subtitle;

        titleBlock.appendChild(titleEl);
        titleBlock.appendChild(subtitleEl);

        // Actions (right)
        const actions = document.createElement('div');
        actions.className = 'card-header-actions';

        // Export button group: SVG · PDF  (CSV added later via addCsvButton)
        const imgBtns = document.createElement('div');
        imgBtns.className = 'card-img-btns';

        ['svg', 'pdf'].forEach(fmt => {
            const imgBtn = document.createElement('button');
            imgBtn.className = 'card-img-btn';
            imgBtn.dataset.fmt = fmt;
            imgBtn.type = 'button';
            imgBtn.title = `Download ${fmt.toUpperCase()}`;
            imgBtn.textContent = fmt.toUpperCase();
            imgBtns.appendChild(imgBtn);
        });

        // Thin separator between export group and action buttons
        const sep = document.createElement('div');
        sep.className = 'card-actions-sep';

        // Expand button
        const expandBtn = document.createElement('button');
        expandBtn.className = 'card-action-btn';
        expandBtn.type = 'button';
        expandBtn.title = 'Expand to fullscreen';
        expandBtn.innerHTML = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><polyline points="15 3 21 3 21 9"/><polyline points="9 21 3 21 3 15"/><line x1="21" y1="3" x2="14" y2="10"/><line x1="3" y1="21" x2="10" y2="14"/></svg>';

        // Delete button
        const deleteBtn = document.createElement('button');
        deleteBtn.className = 'card-action-btn card-delete-btn';
        deleteBtn.type = 'button';
        deleteBtn.title = 'Remove card';
        deleteBtn.innerHTML = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 01-2 2H7a2 2 0 01-2-2V6m3 0V4a2 2 0 012-2h4a2 2 0 012 2v2"/></svg>';

        actions.appendChild(imgBtns);
        actions.appendChild(sep);
        actions.appendChild(expandBtn);
        actions.appendChild(deleteBtn);
        header.appendChild(titleBlock);
        header.appendChild(actions);

        // ── Body ──────────────────────────────────────────────────────────────
        const body = document.createElement('div');
        body.className = 'workspace-card-body';

        const plotContainer = document.createElement('div');
        plotContainer.className = 'plot-container';
        plotContainer.id = cardId;

        body.appendChild(plotContainer);
        card.appendChild(header);
        card.appendChild(body);

        // ── Event listeners ───────────────────────────────────────────────────
        deleteBtn.addEventListener('click', () => {
            const parent = card.parentElement;
            card.remove();
            if (parent && !parent.querySelector('.workspace-card')) {
                const emptyMessages = {
                    visualizationCards: 'No visualizations yet. Use the Explore section on the left to add your first chart.',
                    analysisCards:      'No analysis cards yet. Build a selection and choose “Add to analyze”.',
                    predictionCards:    'No predictions yet. Configure a station, feature, and horizon on the left.',
                    extremeCards:       'No analyses yet. Select a station and feature, then click Run.',
                    climateCards:       'No projections yet. Select a station and feature, then click Generate projection.',
                    changepointCards:   'No analyses yet. Select a station and feature, then click Detect change points.',
                    animateCards:       'No animations yet. Select a dataset and feature, then click Build animation.',
                    scenarioCards:      'No scenario results yet.',
                    mcCards:            'No comparisons yet. Select a station and feature, then click Compare models.',
                    decompCards:        'No decompositions yet. Select a station and feature, then click Decompose series.',
                    waveletCards:       'No analyses yet. Select a station and feature, then click Run wavelet analysis.',
                };
                const msg = emptyMessages[parent.id] || 'No results yet.';
                setEmptyState(parent, msg);
            }
        });
        expandBtn.addEventListener('click', () => openFullscreen(card));
        imgBtns.querySelectorAll('.card-img-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                const plot = card.querySelector('.plot-container');
                if (!plot || !plot.dataset.figure) return;
                if (btn.dataset.fmt === 'pdf') {
                    downloadCardAsPDF(plot, title);
                } else {
                    downloadChartImage(plot, btn.dataset.fmt, title);
                }
            });
        });

        return card;
    }

    function addCsvButton(card, series) {
        card._exportSeries = series;
        const btn = document.createElement('button');
        btn.className = 'card-img-btn';
        btn.dataset.fmt = 'csv';
        btn.type = 'button';
        btn.title = 'Export data as CSV';
        btn.textContent = 'CSV';
        btn.addEventListener('click', () => exportCardCSV(card._exportSeries, card.querySelector('.workspace-card-title')?.textContent || 'export'));
        card.querySelector('.card-img-btns')?.appendChild(btn);
    }

    async function downloadCardAsPDF(plotDiv, title) {
        const jsPDF = window.jspdf?.jsPDF;
        if (!jsPDF) { alert('PDF library not loaded — please refresh and try again.'); return; }
        if (!plotDiv.data) { alert('No chart data to export.'); return; }
        try {
            const imgData = await Plotly.toImage(plotDiv, { format: 'png', width: 1400, height: 700 });
            const pdf = new jsPDF({ orientation: 'landscape', unit: 'pt', format: 'a4' });
            const w = pdf.internal.pageSize.getWidth();
            const h = pdf.internal.pageSize.getHeight();
            pdf.addImage(imgData, 'PNG', 0, 0, w, h);
            const safe = title.replace(/[^a-z0-9_\- ]/gi, '').trim().replace(/\s+/g, '_') || 'chart';
            pdf.save(`${safe}.pdf`);
        } catch (err) {
            alert('PDF export failed: ' + err.message);
        }
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
            const figureJson = plot.dataset.figure;
            if (figureJson) {
                // Clear stale Plotly internals (canvas/WebGL) copied by cloneNode
                plot.innerHTML = '';
                // Use setTimeout so the modal is fully painted before Plotly (especially mapbox) initialises
                setTimeout(() => {
                    renderPlot(plot, figureJson);
                    // After renderPlot's rAF settles, relayout to use the full modal height
                    // so multi-panel figures (e.g. scenario subplots) are not clipped.
                    setTimeout(() => {
                        const modalHeight = Math.floor(window.innerHeight * 0.72);
                        const figure = figureJson ? JSON.parse(figureJson) : null;
                        const originalHeight = Number(figure?.layout?.height || 0);
                        if (plot.data && originalHeight > 0 && modalHeight > originalHeight) {
                            plot.style.height = `${modalHeight}px`;
                            plot.style.minHeight = `${modalHeight}px`;
                            Plotly.relayout(plot, { height: modalHeight }).catch(() => {});
                        }
                    }, 200);
                }, 80);
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

    function pickEvenTicks(values, targetCount) {
        if (!Array.isArray(values) || values.length === 0) return [];
        const count = Math.min(targetCount, values.length);
        if (count === 1) return [values[0]];
        const picks = [];
        const seen = new Set();
        for (let i = 0; i < count; i++) {
            const idx = Math.round(i * (values.length - 1) / (count - 1));
            const val = values[idx];
            if (!seen.has(val)) {
                seen.add(val);
                picks.push(val);
            }
        }
        return picks;
    }

    function applySeasonalTickPolicy(container) {
        const years = container._seasonalYears;
        if (!Array.isArray(years) || years.length === 0) return;
        if (!container._fullLayout) return;
        const width = container.getBoundingClientRect().width;
        const target = width < 520 ? 2 : 4;
        const tickvals = pickEvenTicks(years, target);
        const ticktext = tickvals.map(String);
        const update = {};
        Object.keys(container._fullLayout).forEach((key) => {
            if (!key.startsWith('xaxis')) return;
            update[`${key}.tickmode`] = 'array';
            update[`${key}.tickvals`] = tickvals;
            update[`${key}.ticktext`] = ticktext;
        });
        if (Object.keys(update).length > 0) {
            Plotly.relayout(container, update);
        }
    }

    function renderPlot(container, figureJson) {
        container.dataset.figure = figureJson;
        // Defer to next animation frame so the browser finishes layout before
        // Plotly measures the container width (fixes clipping in card view).
        requestAnimationFrame(() => {
            const figure = JSON.parse(figureJson);
            stripPlotlyUids(figure);
            const layoutHeight = Number(figure?.layout?.height || 0);
            if (layoutHeight > 0) {
                container.style.height = `${layoutHeight}px`;
                container.style.minHeight = `${layoutHeight}px`;
            } else {
                container.style.height = '';
                container.style.minHeight = '';
            }
            if (container.dataset.graphType === 'Calendar Heatmap') {
                figure.layout = figure.layout || {};
                figure.layout.height = Math.max(Number(figure.layout.height || 0), 620);
            }
            const meta = figure?.layout?.meta;
            if (meta?.graph_type === 'Seasonal Subseries Plot') {
                container._seasonalYears = meta.seasonal_years || [];
            } else {
                container._seasonalYears = null;
            }
            const config = {
                responsive: true,
                displaylogo: false,
                modeBarButtonsToRemove: ['lasso2d', 'select2d', 'autoScale2d'],
            };
            Plotly.newPlot(container, figure.data, figure.layout, config).then(() => {
                if (figure.frames && figure.frames.length > 0) {
                    Plotly.addFrames(container, figure.frames);
                }
                applySeasonalTickPolicy(container);
                // Attach a ResizeObserver if not already present
                if (!container._resizeObserver) {
                    container._resizeObserver = new ResizeObserver(() => {
                        if (container.data && container.layout) {
                            // Clear existing width to allow autosizing to the new container width
                            const updateLayout = { ...container.layout };
                            delete updateLayout.width;
                            Plotly.react(container, container.data, updateLayout, config);
                        } else {
                            Plotly.Plots.resize(container);
                        }
                        applySeasonalTickPolicy(container);
                    });
                    container._resizeObserver.observe(container);
                }
            });
        });
    }

    function refreshPlotGrid(container) {
        if (!container) return;
        const config = {
            responsive: true,
            displaylogo: false,
            modeBarButtonsToRemove: ['lasso2d', 'select2d', 'autoScale2d'],
        };
        const resizePlots = () => {
            container.querySelectorAll('.plot-container[data-figure]').forEach((plot) => {
                if (!plot.data || !plot.layout) {
                    renderPlot(plot, plot.dataset.figure);
                    return;
                }
                const originalFigure = plot.dataset.figure ? JSON.parse(plot.dataset.figure) : null;
                const originalHeight = Number(originalFigure?.layout?.height || plot.layout?.height || 0);
                if (originalHeight > 0) {
                    plot.style.height = `${originalHeight}px`;
                    plot.style.minHeight = `${originalHeight}px`;
                }
                const updateLayout = { ...plot.layout, autosize: true };
                if (originalHeight > 0) updateLayout.height = originalHeight;
                delete updateLayout.width;
                Plotly.react(plot, plot.data, updateLayout, config).then(() => {
                    Plotly.Plots.resize(plot);
                    applySeasonalTickPolicy(plot);
                });
            });
        };
        requestAnimationFrame(() => {
            resizePlots();
            setTimeout(resizePlots, 80);
            setTimeout(resizePlots, 220);
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
        // Show loading state immediately, then fetch asynchronously
        els.stationIndicesSection.innerHTML = `
            <div class="indices-loading">
                <span class="indices-spinner"></span> Computing hydro-status…
            </div>`;
        els.stationDetailCard.classList.remove('hidden');
        fetchAndRenderIndices(stationMeta.station);
    }

    async function fetchAndRenderIndices(stationName) {
        try {
            const res = await fetch(`/api/indices?station=${encodeURIComponent(stationName)}`);
            const data = await res.json();
            // Guard: card may have been closed / switched to another station
            if (state.focusedStation !== stationName) return;
            if (!data.ok) throw new Error(data.error);
            renderIndices(els.stationIndicesSection, data.result);
            updateMarkerAlertLevel(stationName, data.result.worst_level);
        } catch (_) {
            if (state.focusedStation === stationName) {
                els.stationIndicesSection.innerHTML =
                    '<div class="indices-empty">Hydro-status unavailable for this station.</div>';
            }
        }
    }

    function renderIndices(container, result) {
        const { spi, flow, worst_level } = result;
        if (!spi && !flow) {
            container.innerHTML = '<div class="indices-empty">No index data available for this station.</div>';
            return;
        }

        const levelMeta = {
            normal:  { bg: 'rgba(34,197,94,0.12)',  border: 'rgba(34,197,94,0.28)',  text: '#4ade80', dot: '#22c55e', badge: '#16a34a' },
            watch:   { bg: 'rgba(251,191,36,0.12)', border: 'rgba(251,191,36,0.3)',  text: '#fbbf24', dot: '#f59e0b', badge: '#b45309' },
            warning: { bg: 'rgba(249,115,22,0.12)', border: 'rgba(249,115,22,0.3)',  text: '#fb923c', dot: '#f97316', badge: '#c2410c' },
            critical:{ bg: 'rgba(239,68,68,0.12)',  border: 'rgba(239,68,68,0.3)',   text: '#f87171', dot: '#ef4444', badge: '#b91c1c' },
        };
        const wm = levelMeta[worst_level] || levelMeta.normal;
        const worstLabel = { normal: 'Normal', watch: 'Watch', warning: 'Warning', critical: 'Critical' }[worst_level] || worst_level;

        let html = `
            <div class="indices-header">
                <span class="indices-title">Hydro-Status</span>
                <span class="alert-badge" style="background:${wm.badge};color:#fff;">${worstLabel.toUpperCase()}</span>
            </div>`;

        if (spi) {
            const m = levelMeta[spi.level] || levelMeta.normal;
            const sign = spi.value >= 0 ? '+' : '';
            html += `
            <div class="index-row" style="background:${m.bg};border-color:${m.border};">
                <div class="index-row-main">
                    <span class="index-dot" style="background:${m.dot};"></span>
                    <div class="index-info">
                        <div class="index-name">SPI-${spi.scale} · ${prettyFeature(spi.feature)}</div>
                        <div class="index-sub">${spi.label} · as of ${spi.latest_date}</div>
                    </div>
                </div>
                <div class="index-value" style="color:${m.text};">${sign}${spi.value.toFixed(2)}</div>
            </div>`;
        }

        if (flow) {
            const m = levelMeta[flow.level] || levelMeta.normal;
            const sign = flow.anomaly_pct >= 0 ? '+' : '';
            const trendIcon = { rising: '↑', falling: '↓', stable: '→' }[flow.trend] || '→';
            html += `
            <div class="index-row" style="background:${m.bg};border-color:${m.border};">
                <div class="index-row-main">
                    <span class="index-dot" style="background:${m.dot};"></span>
                    <div class="index-info">
                        <div class="index-name">Flow Anomaly · ${prettyFeature(flow.feature)}</div>
                        <div class="index-sub">${flow.label} · ${flow.current_value.toFixed(1)} ${flow.unit} · P${flow.percentile.toFixed(0)} · ${trendIcon} ${flow.trend}</div>
                    </div>
                </div>
                <div class="index-value" style="color:${m.text};">${sign}${flow.anomaly_pct.toFixed(1)}%</div>
            </div>`;
        }

        container.innerHTML = html;
    }

    function updateMarkerAlertLevel(stationName, level) {
        const stationMeta = state.stationsByName.get(stationName);
        if (stationMeta) stationMeta._alertLevel = level;
        const marker = state.markers.get(stationName);
        if (marker) marker.setStyle(markerStyleForStation(stationName));
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

    const ALERT_BORDER = { watch: '#f59e0b', warning: '#f97316', critical: '#ef4444' };

    function markerStyleForStation(stationName) {
        const station = state.stationsByName.get(stationName);
        const isFocused = state.focusedStation === stationName;
        const featureCount = station?.features?.length || 1;
        const isLamaH = station?.dataset === 'lamah';
        if (isFocused) {
            return { radius: 9, color: '#ef4444', weight: 2.2, fillColor: '#ef4444', fillOpacity: 0.95 };
        }
        const alertBorder = ALERT_BORDER[station?._alertLevel];
        if (isLamaH) {
            return {
                radius: alertBorder ? 4 : 3,
                color: alertBorder || '#f54842',
                weight: alertBorder ? 2 : 0.5,
                fillColor: LAMAH_COLORS[featureCount] || '#f54842',
                fillOpacity: 0.5,
            };
        }
        return {
            radius: alertBorder ? 7 : 6,
            color: alertBorder || '#e2e8f0',
            weight: alertBorder ? 2.5 : 1.2,
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

    function applyAnalysisMode() {
        const mode = state.analysisMode;
        els.analysisModeSwitch.querySelectorAll('.mode-seg-btn').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.mode === mode);
        });
        const visualizePanel = document.querySelector('[data-sidebar-panel="visualize"]');
        const analysisPanel  = document.querySelector('[data-sidebar-panel="analysis"]');
        if (visualizePanel) visualizePanel.classList.toggle('hidden', mode !== 'charted');
        if (analysisPanel)  analysisPanel.classList.toggle('hidden', mode !== 'free');
    }

    function activateDockTab(tabName) {
        els.dockTabs.forEach((tab) => tab.classList.toggle('active', tab.dataset.dockTab === tabName));
        els.dockPanels.forEach((panel) => panel.classList.toggle('active', panel.dataset.dockPanel === tabName));
        els.dockClearBtns.forEach((btn) => btn.classList.toggle('hidden', btn.dataset.dockClear !== tabName));

        // Show/hide analysis mode switcher
        els.analysisModeSwitch.classList.toggle('hidden', tabName !== 'analysis');

        // Update rail active state
        document.querySelectorAll('.rail-nav-btn').forEach(b => b.classList.toggle('active', b.dataset.railTab === tabName));

        // Toggle sidebar panels
        document.querySelectorAll('[data-sidebar-panel]').forEach(panel => {
            const key = panel.dataset.sidebarPanel;
            if (key === 'visualize') {
                // In analysis tab: visibility controlled by mode switcher
                if (tabName === 'analysis') {
                    panel.classList.toggle('hidden', state.analysisMode !== 'charted');
                } else {
                    panel.classList.toggle('hidden', tabName === 'prediction' || tabName === 'compare' || tabName === 'network' || tabName === 'scenario' || tabName === 'quality' || tabName === 'extreme' || tabName === 'risk' || tabName === 'climate' || tabName === 'changepoint' || tabName === 'animate' || tabName === 'modelcompare' || tabName === 'decompose' || tabName === 'wavelet');
                }
            } else if (key === 'analysis') {
                panel.classList.toggle('hidden', tabName !== 'analysis' || state.analysisMode !== 'free');
            } else if (key === 'prediction') {
                panel.classList.toggle('hidden', tabName !== 'prediction');
            } else if (key === 'compare') {
                panel.classList.toggle('hidden', tabName !== 'compare');
            } else if (key === 'quality') {
                panel.classList.toggle('hidden', tabName !== 'quality');
            } else if (key === 'scenario') {
                panel.classList.toggle('hidden', tabName !== 'scenario');
            } else if (key === 'network') {
                panel.classList.toggle('hidden', tabName !== 'network');
            } else if (key === 'extreme') {
                panel.classList.toggle('hidden', tabName !== 'extreme');
            } else if (key === 'risk') {
                panel.classList.toggle('hidden', tabName !== 'risk');
            } else if (key === 'climate') {
                panel.classList.toggle('hidden', tabName !== 'climate');
            } else if (key === 'changepoint') {
                panel.classList.toggle('hidden', tabName !== 'changepoint');
            } else if (key === 'animate') {
                panel.classList.toggle('hidden', tabName !== 'animate');
            } else if (key === 'modelcompare') {
                panel.classList.toggle('hidden', tabName !== 'modelcompare');
            } else if (key === 'decompose') {
                panel.classList.toggle('hidden', tabName !== 'decompose');
            } else if (key === 'wavelet') {
                panel.classList.toggle('hidden', tabName !== 'wavelet');
            }
        });

        if (tabName === 'analysis') applyAnalysisMode();

        // Toggle action buttons within the explore panel
        document.querySelectorAll('[data-sidebar-action]').forEach(row => {
            const key = row.dataset.sidebarAction;
            if (key === 'visualize') {
                row.classList.toggle('hidden', tabName !== 'visualize');
            } else if (key === 'analysis') {
                row.classList.toggle('hidden', tabName !== 'analysis');
            } else if (key === 'climate') {
                row.classList.toggle('hidden', tabName !== 'climate');
            } else if (key === 'changepoint') {
                row.classList.toggle('hidden', tabName !== 'changepoint');
            } else if (key === 'animate') {
                row.classList.toggle('hidden', tabName !== 'animate');
            } else if (key === 'modelcompare') {
                row.classList.toggle('hidden', tabName !== 'modelcompare');
            } else if (key === 'decompose') {
                row.classList.toggle('hidden', tabName !== 'decompose');
            } else if (key === 'wavelet') {
                row.classList.toggle('hidden', tabName !== 'wavelet');
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

    function capDate(dateStr) {
        const today = new Date().toISOString().split('T')[0];
        return dateStr > today ? today : dateStr;
    }

    function attachDateEnforcement(row, startClass, endClass) {
        function enforce(e) {
            if (e.target.type !== 'date') return;
            const input = e.target;
            const startEl = row.querySelector(startClass);
            const endEl   = row.querySelector(endClass);
            if (!startEl || !endEl) return;
            // Clamp individual input to its own min/max
            if (input.min && input.value < input.min) input.value = input.min;
            if (input.max && input.value > input.max) input.value = input.max;
            // If start is after end for any reason, always reset end to its max
            if (startEl.type === 'date' && endEl.type === 'date' && startEl.value > endEl.value) {
                endEl.value = endEl.max || endEl.value;
            }
        }
        row.addEventListener('change', enforce);
    }

    function prettyStation(name) {
        const meta = state.stationsByName.get(name);
        if (meta?.name && meta.name !== name) return meta.name.replace(/_/g, ' ');
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

    // ── Export helpers ─────────────────────────────────────────────────────

    function downloadChartImage(plotDiv, format, title) {
        const safe = title.replace(/[^a-z0-9_\- ]/gi, '').trim().replace(/\s+/g, '_') || 'chart';
        Plotly.downloadImage(plotDiv, { format, filename: safe, width: 1400, height: 700 });
    }

    async function exportCardCSV(series, title) {
        try {
            const res = await fetch('/api/export-csv', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ series }),
            });
            if (!res.ok) {
                const err = await res.json();
                alert('CSV export failed: ' + (err.error || res.statusText));
                return;
            }
            const blob = await res.blob();
            const safe = title.replace(/[^a-z0-9_\- ]/gi, '').trim().replace(/\s+/g, '_') || 'export';
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `${safe}.csv`;
            a.click();
            URL.revokeObjectURL(url);
        } catch (err) {
            alert('CSV export failed: ' + err.message);
        }
    }

    async function generateSessionPDF() {
        const jsPDF = window.jspdf?.jsPDF;
        if (!jsPDF) { alert('PDF library not loaded — please refresh and try again.'); return; }

        const btn = els.exportPdfBtn;
        const originalText = btn.textContent;
        btn.disabled = true;
        btn.textContent = 'Building PDF…';

        try {
            const doc = new jsPDF({ orientation: 'portrait', unit: 'mm', format: 'a4' });
            const PW = 210, PH = 297, M = 14, CW = PW - M * 2;

            // ── Title page ──────────────────────────────────────────────
            doc.setFillColor(15, 23, 42);
            doc.rect(0, 0, PW, PH, 'F');
            doc.setTextColor(248, 250, 252);
            doc.setFont('helvetica', 'bold');
            doc.setFontSize(30);
            doc.text('HydroVision', PW / 2, 90, { align: 'center' });
            doc.setFont('helvetica', 'normal');
            doc.setFontSize(13);
            doc.setTextColor(148, 163, 184);
            doc.text('Session Report', PW / 2, 103, { align: 'center' });
            doc.setFontSize(10);
            doc.text(new Date().toLocaleString(), PW / 2, 114, { align: 'center' });

            const panelInfo = [
                { id: 'visualizationCards',  label: 'Visualization' },
                { id: 'analysisCards',       label: 'Analysis' },
                { id: 'predictionCards',     label: 'Prediction' },
                { id: 'scenarioCards',       label: 'Scenario' },
                { id: 'extremeCards',        label: 'Extreme Events' },
                { id: 'climateCards',        label: 'Climate Projection' },
                { id: 'changepointCards',    label: 'Change Points' },
                { id: 'mcCards',             label: 'Model Comparison' },
                { id: 'decompCards',         label: 'STL Decomposition' },
                { id: 'waveletCards',        label: 'Wavelet Analysis' },
            ];
            const counts = panelInfo.map(p => document.getElementById(p.id)?.querySelectorAll('.workspace-card').length ?? 0);
            doc.setFontSize(10);
            doc.setTextColor(100, 116, 139);
            doc.text(
                panelInfo.map((p, i) => `${counts[i]} ${p.label}`).join('  ·  '),
                PW / 2, 128, { align: 'center' }
            );

            // ── One page per card ────────────────────────────────────────
            for (const { id, label } of panelInfo) {
                const cards = Array.from(document.getElementById(id)?.querySelectorAll('.workspace-card') ?? []);
                for (const card of cards) {
                    const title    = card.querySelector('.workspace-card-title')?.textContent ?? '';
                    const subtitle = card.querySelector('.workspace-card-subtitle')?.textContent ?? '';
                    doc.addPage();
                    doc.setFillColor(255, 255, 255);
                    doc.rect(0, 0, PW, PH, 'F');

                    // Section badge
                    doc.setFillColor(226, 232, 240);
                    doc.roundedRect(M, M, 22, 6, 1.5, 1.5, 'F');
                    doc.setFontSize(7);
                    doc.setTextColor(71, 85, 105);
                    doc.setFont('helvetica', 'bold');
                    doc.text(label.toUpperCase(), M + 11, M + 4.3, { align: 'center' });

                    // Title
                    doc.setFont('helvetica', 'bold');
                    doc.setFontSize(14);
                    doc.setTextColor(15, 23, 42);
                    doc.text(title, M, M + 14);

                    // Subtitle (wrap)
                    doc.setFont('helvetica', 'normal');
                    doc.setFontSize(8);
                    doc.setTextColor(100, 116, 139);
                    const subLines = doc.splitTextToSize(subtitle, CW);
                    doc.text(subLines, M, M + 21);

                    let yPos = M + 21 + subLines.length * 4 + 4;

                    // Charts
                    const plots = Array.from(card.querySelectorAll('.plot-container')).filter(p => p.dataset.figure);
                    for (const plot of plots) {
                        try {
                            const imgData = await window.Plotly.toImage(plot, { format: 'png', width: 1400, height: 620 });
                            const imgH = CW * 620 / 1400;
                            if (yPos + imgH > PH - M) { doc.addPage(); doc.setFillColor(255,255,255); doc.rect(0,0,PW,PH,'F'); yPos = M; }
                            doc.addImage(imgData, 'PNG', M, yPos, CW, imgH);
                            yPos += imgH + 6;
                        } catch (_) { /* skip unrendered plot */ }
                    }

                    // Analysis narrative text
                    const summaryEl = card.querySelector('.analysis-summary');
                    if (summaryEl?.innerText?.trim()) {
                        const text = summaryEl.innerText.trim();
                        doc.setFont('helvetica', 'normal');
                        doc.setFontSize(8.5);
                        doc.setTextColor(30, 41, 59);
                        const lines = doc.splitTextToSize(text, CW);
                        let li = 0;
                        while (li < lines.length) {
                            const chunk = lines.slice(li, li + 35);
                            if (yPos + chunk.length * 4.2 > PH - M) { doc.addPage(); doc.setFillColor(255,255,255); doc.rect(0,0,PW,PH,'F'); yPos = M; }
                            doc.text(chunk, M, yPos);
                            yPos += chunk.length * 4.2 + 2;
                            li += 35;
                        }
                    }
                }
            }

            const dateStr = new Date().toISOString().slice(0, 10);
            doc.save(`hydrovision_report_${dateStr}.pdf`);
        } finally {
            btn.disabled = false;
            btn.textContent = originalText;
        }
    }

    // ── Compare workspace ──────────────────────────────────────────────────

    function updateCompareYearHint() {
        if (!els.compareYearHint || !state.bootstrap) return;
        const ds = state.compareDataset;
        const feature = els.compareFeatureSelect?.value;
        if (!feature) { els.compareYearHint.textContent = ''; return; }

        let minYear = Infinity, maxYear = -Infinity;
        state.bootstrap.stations.forEach(s => {
            if (s.dataset !== ds) return;
            const fd = s.feature_details?.[feature];
            if (!fd) return;
            const sy = new Date(fd.start_date).getFullYear();
            const ey = new Date(fd.end_date).getFullYear();
            if (sy < minYear) minYear = sy;
            if (ey > maxYear) maxYear = ey;
        });

        if (minYear === Infinity) { els.compareYearHint.textContent = ''; return; }

        els.compareYearInput.min = minYear;
        els.compareYearInput.max = maxYear;
        els.compareYearHint.textContent = `Available range: ${minYear} – ${maxYear}`;

        // Clear invalid input when range changes
        const currentVal = parseInt(els.compareYearInput.value, 10);
        if (!isNaN(currentVal) && (currentVal < minYear || currentVal > maxYear)) {
            els.compareYearInput.value = '';
        }
    }

    function populateCompareFeatures() {
        if (!els.compareFeatureSelect || !state.bootstrap) return;
        const ds = state.compareDataset;
        const datasetFeatures = state.bootstrap.dataset_features || {};
        const features = datasetFeatures[ds] || state.bootstrap.features || [];
        els.compareFeatureSelect.innerHTML = '';
        features.forEach(f => {
            const opt = document.createElement('option');
            opt.value = f;
            opt.textContent = f.replaceAll('_', ' ');
            els.compareFeatureSelect.appendChild(opt);
        });
        updateCompareYearHint();
    }

    async function runComparison() {
        const feature = els.compareFeatureSelect?.value;
        if (!feature) {
            showMessage(els.compareMessage, 'Please select a feature.', 'error');
            return;
        }
        const yearRaw = els.compareYearInput?.value?.trim();
        const year = yearRaw ? parseInt(yearRaw, 10) : null;
        const component = els.compareComponentSelect?.value ?? 'all';

        // Validate year is within range if provided
        if (year) {
            const minYear = parseInt(els.compareYearInput.min, 10);
            const maxYear = parseInt(els.compareYearInput.max, 10);
            if (isNaN(year) || year < minYear || year > maxYear) {
                showMessage(els.compareMessage, `Year must be between ${minYear} and ${maxYear}.`, 'error');
                return;
            }
        }

        const includeAnalysis = document.getElementById('compareAnalysisToggle')?.checked ?? false;
        showMessage(els.compareMessage, 'Running comparison…', '');
        els.runCompareBtn.disabled = true;

        try {
            const res = await fetch('/api/compare', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ dataset: state.compareDataset, feature, year, component, include_analysis: includeAnalysis }),
            });
            const data = await res.json();
            if (!res.ok || !data.ok) throw new Error(data.error || 'Comparison failed.');

            activateDockTab('compare');
            renderCompareResult(data.result, component);
            showMessage(els.compareMessage, 'Done.', 'success');
        } catch (err) {
            showMessage(els.compareMessage, err.message || 'Something went wrong.', 'error');
        } finally {
            els.runCompareBtn.disabled = false;
        }
    }

    function renderCompareResult(result, component) {
        if (!els.compareWorkspace) return;
        els.compareWorkspace.innerHTML = '';

        if (component === 'all') {
            if (result.errors?.length) {
                const errDiv = document.createElement('div');
                errDiv.className = 'compare-errors';
                errDiv.innerHTML = result.errors.map(e => `<div class="compare-error-item">${escapeHtml(e)}</div>`).join('');
                els.compareWorkspace.appendChild(errDiv);
            }
            if (result.correlation) renderCorrelationMatrix(result.correlation);
            if (result.leaderboard) renderAnomalyLeaderboard(result.leaderboard);
            if (result.summary) renderBasinSummary(result.summary);
        } else if (component === 'correlation') {
            renderCorrelationMatrix(result);
        } else if (component === 'leaderboard') {
            renderAnomalyLeaderboard(result);
        } else if (component === 'summary') {
            renderBasinSummary(result);
        }
    }

    function appendCompareAnalysis(section, analysisText) {
        if (!section || !analysisText) return;
        const block = document.createElement('div');
        block.className = 'scenario-analysis-section compare-analysis-section';
        const looksLikeHtml = /<\/?(p|ul|li|h1|h2|h3|h4|strong|em|br)\b/i.test(analysisText);
        const formatted = looksLikeHtml
            ? analysisText
            : escapeHtml(analysisText)
                .replace(/\n/g, '<br>')
                .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
        block.innerHTML = `
            <div class="analysis-header">
                <h4>Analysis Report</h4>
            </div>
            <div class="analysis-content">${formatted}</div>
        `;
        const header = section.querySelector('.compare-section-header');
        if (header && header.nextSibling) {
            section.insertBefore(block, header.nextSibling);
        } else {
            section.appendChild(block);
        }
    }

    function renderCorrelationMatrix(data) {
        const section = document.createElement('div');
        section.className = 'compare-section';

        const header = `
            <div class="compare-section-header">
                <h4>Correlation Matrix — ${escapeHtml(data.feature.replaceAll('_', ' '))} (${escapeHtml(data.dataset)})</h4>
                <span class="compare-section-meta">${data.n_stations} stations${data.capped ? ` (capped from ${data.total_available})` : ''}</span>
            </div>`;

        // Build heatmap via Plotly
        const plotDiv = document.createElement('div');
        plotDiv.className = 'compare-heatmap-plot';
        section.innerHTML = header;
        if (data.analysis) {
            appendCompareAnalysis(section, data.analysis);
        }
        section.appendChild(plotDiv);
        els.compareWorkspace.appendChild(section);

        const labels = data.stations;
        const matrix = data.matrix;

        Plotly.newPlot(plotDiv, [{
            type: 'heatmap',
            z: matrix,
            x: labels,
            y: labels,
            colorscale: 'RdBu',
            zmin: -1, zmax: 1,
            showscale: true,
            colorbar: { x: 1.01, thickness: 14, tickfont: { size: 9 } },
            hoverongaps: false,
            hovertemplate: '%{y} × %{x}: <b>%{z:.3f}</b><extra></extra>',
        }], {
            paper_bgcolor: 'transparent',
            plot_bgcolor: 'transparent',
            font: { color: 'var(--text)', size: 10 },
            margin: { t: 20, b: 120, l: 120, r: 90 },
            xaxis: { tickangle: -45, tickfont: { size: 9 } },
            yaxis: { tickfont: { size: 9 } },
            height: Math.max(320, Math.min(600, data.n_stations * 12 + 140)),
        }, { responsive: true, displayModeBar: false }).then(() => {
            if (!plotDiv._resizeObserver) {
                plotDiv._resizeObserver = new ResizeObserver(() => Plotly.Plots.resize(plotDiv));
                plotDiv._resizeObserver.observe(plotDiv);
            }
        });

        // Mean correlation table
        const sortedMean = data.station_ids
            .map((id, i) => ({ id, name: data.stations[i], mean: data.mean_correlations[i] }))
            .filter(r => r.mean !== null)
            .sort((a, b) => b.mean - a.mean);

        const tableWrap = document.createElement('div');
        tableWrap.className = 'compare-table-wrap';
        tableWrap.innerHTML = `
            <div class="compare-subsection-title">Mean pairwise correlation per station</div>
            <table class="compare-table">
                <thead><tr><th>Station</th><th>Mean r</th><th>Bar</th></tr></thead>
                <tbody>${sortedMean.slice(0, 20).map(r => {
                    const pct = Math.round(Math.abs(r.mean) * 100);
                    const color = r.mean >= 0 ? '#34d399' : '#f87171';
                    return `<tr>
                        <td>${escapeHtml(r.name.replace(/_/g, ' '))}</td>
                        <td class="compare-num ${r.mean >= 0 ? 'pos' : 'neg'}">${r.mean.toFixed(3)}</td>
                        <td><div class="compare-bar-bg"><div class="compare-bar-fill" style="width:${pct}%;background:${color}"></div></div></td>
                    </tr>`;
                }).join('')}</tbody>
            </table>`;
        section.appendChild(tableWrap);
    }

    const LEVEL_COLORS = { normal: '#64748b', watch: '#f59e0b', warning: '#f97316', critical: '#ef4444' };

    function renderAnomalyLeaderboard(data) {
        const section = document.createElement('div');
        section.className = 'compare-section';

        const featureLabel = data.feature.replaceAll('_', ' ');
        const unitStr = data.unit ? ` (${escapeHtml(data.unit)})` : '';
        section.innerHTML = `
            <div class="compare-section-header">
                <h4>Anomaly Leaderboard — ${escapeHtml(featureLabel)}${unitStr} · ${data.year ?? '—'}</h4>
                <span class="compare-section-meta">${data.total_stations} stations · ${data.above_normal} above / ${data.below_normal} below normal</span>
            </div>
            <div class="compare-leaderboard">
                ${data.rows.map((r, i) => {
                    const color = LEVEL_COLORS[r.level] ?? '#64748b';
                    const arrow = r.direction === 'above' ? '▲' : '▼';
                    const sign = r.anomaly_pct >= 0 ? '+' : '';
                    const barPct = Math.min(100, Math.abs(r.anomaly_pct));
                    return `<div class="lb-row">
                        <span class="lb-rank">${i + 1}</span>
                        <div class="lb-info">
                            <span class="lb-name">${escapeHtml((r.name || r.station).replace(/_/g, ' '))}</span>
                            <div class="lb-bar-bg"><div class="lb-bar-fill" style="width:${barPct}%;background:${color}"></div></div>
                        </div>
                        <span class="lb-pct" style="color:${color}">${arrow} ${sign}${r.anomaly_pct.toFixed(1)}%</span>
                        <span class="lb-badge" style="background:${color}20;color:${color};border:1px solid ${color}40">${r.level}</span>
                    </div>`;
                }).join('')}
            </div>`;
        if (data.analysis) {
            appendCompareAnalysis(section, data.analysis);
        }
        els.compareWorkspace.appendChild(section);
    }

    function renderBasinSummary(data) {
        const section = document.createElement('div');
        section.className = 'compare-section';

        const featureLabel = data.feature.replaceAll('_', ' ');
        const u = data.unit ? ` ${escapeHtml(data.unit)}` : '';
        const fmt = v => (v == null ? '—' : Number(v).toFixed(3));

        const chips = [
            { label: 'Mean', value: `${fmt(data.basin_mean)}${u}` },
            { label: 'Median', value: `${fmt(data.basin_median)}${u}` },
            { label: 'Std dev', value: `${fmt(data.basin_std)}${u}` },
            { label: 'Min', value: `${fmt(data.basin_min)}${u}` },
            { label: 'Max', value: `${fmt(data.basin_max)}${u}` },
            { label: 'CV', value: `${data.spatial_cv_pct}%` },
            { label: 'P10', value: `${fmt(data.p10)}${u}` },
            { label: 'P25', value: `${fmt(data.p25)}${u}` },
            { label: 'P75', value: `${fmt(data.p75)}${u}` },
            { label: 'P90', value: `${fmt(data.p90)}${u}` },
            { label: 'Stations', value: `${data.active_stations} / ${data.total_stations}` },
            { label: 'Observations', value: data.total_observations?.toLocaleString() ?? '—' },
            { label: 'Imputation', value: `${data.avg_imputation_pct}%` },
            ...(data.trends_computed ? [
                { label: 'Rising', value: data.trends.rising },
                { label: 'Stable', value: data.trends.stable },
                { label: 'Falling', value: data.trends.falling },
            ] : []),
        ].map(c => `<div class="basin-chip"><span class="basin-chip-label">${c.label}</span><span class="basin-chip-value">${c.value}</span></div>`).join('');

        section.innerHTML = `
            <div class="compare-section-header">
                <h4>Basin Summary — ${escapeHtml(featureLabel)} (${escapeHtml(data.dataset)})</h4>
                <span class="compare-section-meta">Highest: ${escapeHtml(data.highest_station.name)} · Lowest: ${escapeHtml(data.lowest_station.name)}</span>
            </div>
            <div class="basin-chips">${chips}</div>`;
        if (data.analysis) {
            appendCompareAnalysis(section, data.analysis);
        }

        // Histogram
        if (data.histogram) {
            const histDiv = document.createElement('div');
            histDiv.className = 'compare-hist-plot';
            section.appendChild(histDiv);
            els.compareWorkspace.appendChild(section);

            const { counts, edges } = data.histogram;
            const xLabels = counts.map((_, i) => ((edges[i] + edges[i + 1]) / 2).toFixed(2));
            Plotly.newPlot(histDiv, [{
                type: 'bar',
                x: xLabels,
                y: counts,
                marker: { color: '#60a5fa', opacity: 0.85 },
                hovertemplate: 'Mean ≈ %{x}<br>Stations: %{y}<extra></extra>',
            }], {
                paper_bgcolor: 'transparent',
                plot_bgcolor: 'transparent',
                font: { color: 'var(--text)', size: 11 },
                margin: { t: 10, b: 50, l: 50, r: 10 },
                height: 200,
                xaxis: { title: { text: `Station mean (${data.unit || 'value'})`, font: { size: 11 } }, gridcolor: 'rgba(148,163,184,0.1)' },
                yaxis: { title: { text: 'Count', font: { size: 11 } }, gridcolor: 'rgba(148,163,184,0.1)' },
                bargap: 0.05,
            }, { responsive: true, displayModeBar: false });
        } else {
            els.compareWorkspace.appendChild(section);
        }
    }

    function appendCompareBriefing(body, analysisText, meta = {}) {
        if (!analysisText || !body) return;
        const section = document.createElement('section');
        section.className = 'compare-briefing-card';
        const featureLabel = (meta.feature || 'Unknown feature').replaceAll('_', ' ');
        const datasetLabel = (meta.dataset || 'dataset').replace(/_/g, ' ');
        const componentLabel = meta.component === 'all'
            ? 'Full Basin Comparison'
            : `${(meta.component || 'comparison').charAt(0).toUpperCase()}${(meta.component || 'comparison').slice(1)} Focus`;
        section.innerHTML = `
            <div class="compare-briefing-topbar">
                <div class="compare-briefing-kicker">Interpretation</div>
                <div class="compare-briefing-meta">${escapeHtml(featureLabel)} · ${escapeHtml(datasetLabel)} · ${escapeHtml(componentLabel)}</div>
            </div>
            <div class="compare-briefing-body"></div>
        `;

        const bodyEl = section.querySelector('.compare-briefing-body');
        const looksLikeHtml = /<\/?(p|ul|li|h1|h2|h3|h4|strong|em|br)\b/i.test(analysisText);
        if (looksLikeHtml) {
            bodyEl.innerHTML = analysisText;
        } else {
            bodyEl.innerHTML = escapeHtml(analysisText)
                .replace(/\n/g, '<br>')
                .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
        }

        body.appendChild(section);
    }

    // ── Network Analysis ─────────────────────────────────────────────────────

    // Cache full network result so switching views doesn't re-fetch
    let _networkCache = null;

    async function runNetworkAnalysis() {
        if (!els.runNetworkBtn) return;
        const activeBtn = els.networkViewPicker?.querySelector('.mode-seg-btn.active');
        const view = activeBtn?.dataset.view || 'graph';
        const includeAnalysis = document.getElementById('networkAnalysisToggle')?.checked ?? false;
        showMessage(els.networkMessage, 'Loading network data…', '');
        els.runNetworkBtn.disabled = true;

        try {
            // Fetch full network (cached after first call, but re-fetch if analysis requested)
            if (!_networkCache || includeAnalysis) {
                const params = new URLSearchParams({ dataset: 'mekong' });
                if (includeAnalysis) params.set('include_analysis', 'true');
                const res = await fetch(`/api/network?${params}`);
                const data = await res.json();
                if (!data.ok) throw new Error(data.error);
                _networkCache = data.result;
                // Populate contribution station select once
                populateNetworkContribSelect(_networkCache.nodes);
                // Update stat card values directly
                if (els.netChipNodes) els.netChipNodes.textContent = _networkCache.node_count;
                if (els.netChipEdges) els.netChipEdges.textContent = _networkCache.edge_count;
                if (els.netChipStem) els.netChipStem.textContent = _networkCache.main_stem.length;
            }

            activateDockTab('network');
            if (els.networkWorkspace) els.networkWorkspace.innerHTML = '';

            if (view === 'graph') {
                renderNetworkGraph(_networkCache);
            } else if (view === 'travel') {
                renderTravelTimes(_networkCache.travel_times, includeAnalysis ? _networkCache.analysis : null);
            } else if (view === 'contribution') {
                const station = els.networkContribStation?.value;
                if (!station) { showMessage(els.networkMessage, 'Select a target station.', 'error'); return; }
                await renderContribution(station, includeAnalysis);
            }

            showMessage(els.networkMessage, 'Done.', 'success');
        } catch (err) {
            showMessage(els.networkMessage, err.message || 'Network analysis failed.', 'error');
        } finally {
            els.runNetworkBtn.disabled = false;
        }
    }

    function populateNetworkContribSelect(nodes) {
        if (!els.networkContribStation) return;
        const previousValue = els.networkContribStation.value;
        els.networkContribStation.innerHTML = '';
        // Main stem first, then tributaries
        const sorted = [...nodes].sort((a, b) => {
            if (a.main_stem && !b.main_stem) return -1;
            if (!a.main_stem && b.main_stem) return 1;
            return a.name.localeCompare(b.name);
        });
        let currentGroup = null;
        sorted.forEach(n => {
            const group = n.main_stem ? 'Main stem' : 'Tributaries';
            if (!currentGroup || currentGroup.label !== group) {
                currentGroup = document.createElement('optgroup');
                currentGroup.label = group;
                els.networkContribStation.appendChild(currentGroup);
            }
            const opt = document.createElement('option');
            opt.value = n.id;
            opt.textContent = n.name.replace(/_/g, ' ');
            if (n.id === previousValue) opt.selected = true;
            currentGroup.appendChild(opt);
        });
        if (!els.networkContribStation.value && previousValue) {
            const fallback = els.networkContribStation.querySelector(`option[value="${CSS.escape(previousValue)}"]`);
            if (fallback) fallback.selected = true;
        }
    }

    function populateNetworkContribFromBootstrap() {
        if (!els.networkContribStation || !state.bootstrap) return;
        // Only populate if dropdown is currently empty (don't overwrite network-API data)
        if (els.networkContribStation.options.length > 0) return;
        const stations = (state.bootstrap.stations || [])
            .filter(s => s.dataset === 'mekong')
            .sort((a, b) => a.name.localeCompare(b.name));
        if (stations.length === 0) return;
        els.networkContribStation.innerHTML = '';
        const opt0 = document.createElement('option');
        opt0.value = '';
        opt0.textContent = '— select a station —';
        opt0.disabled = true;
        opt0.selected = true;
        els.networkContribStation.appendChild(opt0);
        stations.forEach(s => {
            const opt = document.createElement('option');
            opt.value = s.station;
            opt.textContent = (s.name || s.station).replace(/_/g, ' ');
            els.networkContribStation.appendChild(opt);
        });
    }

    function renderNetworkGraph(result) {
        if (!els.networkWorkspace) return;

        const section = document.createElement('div');
        section.className = 'network-section network-section-graph';

        const header = document.createElement('div');
        header.className = 'network-graph-header';
        header.innerHTML = `
            <div class="network-graph-titlewrap">
                <h4 class="network-section-title">Spatial Network Topology</h4>
                <p class="network-note">${escapeHtml(result.note || '')}</p>
            </div>
            <div class="network-graph-legend" aria-label="Graph legend">
                <span class="network-legend-chip"><span class="network-legend-swatch network-legend-swatch-main"></span>Main stem path</span>
                <span class="network-legend-chip"><span class="network-legend-swatch network-legend-swatch-trib-line"></span>Tributary links</span>
                <span class="network-legend-chip"><span class="network-legend-swatch network-legend-swatch-node-main"></span>Main-stem stations</span>
                <span class="network-legend-chip"><span class="network-legend-swatch network-legend-swatch-node-trib"></span>Tributary stations</span>
                <span class="network-legend-chip"><span class="network-legend-arrow">→</span>Downstream direction</span>
            </div>`;
        section.appendChild(header);

        const plotDiv = document.createElement('div');
        plotDiv.className = 'network-plot-container';
        section.appendChild(plotDiv);

        const guide = document.createElement('div');
        guide.className = 'network-graph-footer';
        guide.innerHTML = `
            <div class="network-graph-guide">${escapeHtml(result.graph_guide || '')}</div>
            <div class="network-graph-caveat">${escapeHtml(result.travel_time_note || '')}</div>`;
        section.appendChild(guide);

        if (result.analysis) {
            appendAnalysisSection(section, result.analysis, { title: 'Analysis Report' });
        }

        els.networkWorkspace.appendChild(section);

        renderPlot(plotDiv, result.figure);
    }

    function renderTravelTimes(rows, analysisText = null) {
        if (!els.networkWorkspace) return;
        const section = document.createElement('div');
        section.className = 'network-section';

        const heading = document.createElement('h4');
        heading.className = 'network-section-title';
        heading.textContent = 'Empirical Travel Times — Main Mekong Stem';
        section.appendChild(heading);

        const caveat = document.createElement('p');
        caveat.className = 'network-caveat-text';
        caveat.textContent = 'Lag at peak cross-correlation of monthly discharge series between consecutive main-stem pairs. Monthly resolution means 1 lag ≈ 30 days.';
        section.appendChild(caveat);

        if (!rows || rows.length === 0) {
            const empty = document.createElement('p');
            empty.className = 'network-caveat-text';
            empty.textContent = 'Insufficient overlapping discharge data for cross-correlation.';
            section.appendChild(empty);
            els.networkWorkspace.appendChild(section);
            return;
        }

        const table = document.createElement('table');
        table.className = 'network-table';
        table.innerHTML = `
            <thead><tr>
                <th>Upstream station</th>
                <th>Downstream station</th>
                <th>Lag (months)</th>
                <th>Correlation</th>
                <th>Overlap (months)</th>
            </tr></thead>`;
        const tbody = document.createElement('tbody');
        rows.forEach(r => {
            const tr = document.createElement('tr');
            const corrClass = r.correlation >= 0.7 ? 'net-corr-high' : r.correlation >= 0.4 ? 'net-corr-mid' : 'net-corr-low';
            tr.innerHTML = `
                <td>${r.upstream_name.replace(/_/g, ' ')}</td>
                <td>${r.downstream_name.replace(/_/g, ' ')}</td>
                <td><strong>${r.lag_months}</strong></td>
                <td class="${corrClass}">${r.correlation.toFixed(3)}</td>
                <td>${r.overlap_months}</td>`;
            tbody.appendChild(tr);
        });
        table.appendChild(tbody);
        section.appendChild(table);

        if (analysisText) {
            appendAnalysisSection(section, analysisText, { title: 'Analysis Report' });
        }

        els.networkWorkspace.appendChild(section);
    }

    async function renderContribution(station, includeAnalysis = false) {
        let url = `/api/network/contribution?station=${encodeURIComponent(station)}&dataset=mekong`;
        if (includeAnalysis) url += '&include_analysis=true';
        const res = await fetch(url);
        const data = await res.json();
        if (!data.ok) throw new Error(data.error);
        const result = data.result;

        const section = document.createElement('div');
        section.className = 'network-section';

        const heading = document.createElement('h4');
        heading.className = 'network-section-title';
        heading.textContent = `Upstream Discharge Contributions → ${result.target_name.replace(/_/g, ' ')}`;
        section.appendChild(heading);

        const meta = document.createElement('p');
        meta.className = 'network-caveat-text';
        meta.textContent = `Target mean discharge: ${result.target_mean_q !== null ? result.target_mean_q + ' m³/s' : 'N/A'} · ${result.upstream_count} upstream stations found.`;
        section.appendChild(meta);

        if (result.rows.length === 0) {
            const empty = document.createElement('p');
            empty.className = 'network-caveat-text';
            empty.textContent = 'No upstream stations found in topology.';
            section.appendChild(empty);
        } else {
            const table = document.createElement('table');
            table.className = 'network-table';
            table.innerHTML = `
                <thead><tr>
                    <th>Station</th>
                    <th>Country</th>
                    <th>Mean discharge (m³/s)</th>
                    <th>Contribution (proxy %)</th>
                </tr></thead>`;
            const tbody = document.createElement('tbody');
            result.rows.forEach(r => {
                const tr = document.createElement('tr');
                const pct = r.contrib_pct !== null ? r.contrib_pct.toFixed(1) + '%' : '—';
                const bar = r.contrib_pct !== null
                    ? `<div class="net-contrib-bar" style="width:${Math.min(r.contrib_pct, 100)}%"></div>`
                    : '';
                tr.innerHTML = `
                    <td>${r.name.replace(/_/g, ' ')}</td>
                    <td>${r.country}</td>
                    <td>${r.mean_q !== null ? r.mean_q : '—'}</td>
                    <td><div class="net-contrib-cell">${bar}<span>${pct}</span></div></td>`;
                tbody.appendChild(tr);
            });
            table.appendChild(tbody);
            section.appendChild(table);
        }

        const caveat = document.createElement('p');
        caveat.className = 'network-caveat-text network-caveat-small';
        caveat.textContent = result.caveat;
        section.appendChild(caveat);

        if (result.analysis) {
            appendAnalysisSection(section, result.analysis, { title: 'Interpretation' });
        }

        els.networkWorkspace.appendChild(section);
    }

    // ── Scenario ─────────────────────────────────────────────────────────────

    function populateScenarioControls() {
        if (!els.scenarioStationSelect) return;
        // Populate station select — Mekong only (needs multi-feature stations for cross-feature sensitivity)
        const stations = Object.values(state.bootstrap?.stations || [])
            .filter(s => s.dataset === 'mekong' && s.features.length >= 1)
            .sort((a, b) => a.name.localeCompare(b.name));

        els.scenarioStationSelect.innerHTML = stations
            .map(s => `<option value="${escapeHtml(s.station)}">${escapeHtml(s.name.replace(/_/g, ' '))}</option>`)
            .join('');

        // Populate model select from predict models
        if (els.scenarioModelSelect) {
            const models = Array.from(els.predictModelSelect?.options || []).map(o => o.value);
            els.scenarioModelSelect.innerHTML = models
                .map(m => `<option value="${escapeHtml(m)}"${m === 'FlowNet' ? ' selected' : ''}>${escapeHtml(m)}</option>`)
                .join('');
        }

        updateScenarioFeatureOptions();
    }

    function updateScenarioFeatureOptions() {
        if (!els.scenarioStationSelect || !els.scenarioTargetSelect || !els.scenarioDriverSelect) return;
        const station = els.scenarioStationSelect.value;
        const stInfo = (state.bootstrap?.stations || []).find(s => s.station === station);
        const features = stInfo?.features || [];

        const opts = features.map(f => `<option value="${escapeHtml(f)}">${escapeHtml(f.replace(/_/g, ' '))}</option>`).join('');
        els.scenarioTargetSelect.innerHTML = opts;
        els.scenarioDriverSelect.innerHTML = opts;

        // Default: target=Discharge (or first), driver=Rainfall (or same)
        const setVal = (sel, preferred) => {
            const match = [...sel.options].find(o => o.value === preferred);
            if (match) sel.value = preferred;
        };
        setVal(els.scenarioTargetSelect, 'Discharge');
        setVal(els.scenarioDriverSelect, 'Rainfall');
        if (els.scenarioDriverSelect.value === els.scenarioTargetSelect.value) {
            // If same, try Rainfall or Precipitation
            const fallback = features.find(f => f === 'Rainfall' || f === 'Precipitation');
            if (fallback) els.scenarioDriverSelect.value = fallback;
        }
    }

    async function runScenario() {
        if (!els.runScenarioBtn) return;
        const station = els.scenarioStationSelect?.value;
        const targetFeature = els.scenarioTargetSelect?.value;
        const driverFeature = els.scenarioDriverSelect?.value;
        const scalePct = Number(els.scenarioScaleSlider?.value || 20);
        const durationMonths = Number(els.scenarioDurationSlider?.value || 3);
        const startOffset = Number(els.scenarioOffsetSlider?.value || 0);
        const model = els.scenarioModelSelect?.value || 'FlowNet';
        const includeAnalysis = document.getElementById('scenarioAnalysisToggle')?.checked ?? true;

        if (!station || !targetFeature) {
            showMessage(els.scenarioMessage, 'Select a station and target feature.', 'error');
            return;
        }
        showMessage(els.scenarioMessage, 'Running scenario…', '');
        els.runScenarioBtn.disabled = true;

        try {
            const res = await fetch('/api/scenario', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ station, target_feature: targetFeature, driver_feature: driverFeature,
                    scale_pct: scalePct, duration_months: durationMonths, start_offset: startOffset,
                    model, horizon: 12, include_analysis: includeAnalysis }),
            });
            const data = await res.json();
            if (!data.ok) throw new Error(data.error);
            appendScenarioCard(data.result);
            activateDockTab('scenario');
            showMessage(els.scenarioMessage, 'Done.', 'success');
        } catch (err) {
            showMessage(els.scenarioMessage, err.message || 'Scenario failed.', 'error');
        } finally {
            els.runScenarioBtn.disabled = false;
        }
    }

    function appendScenarioCard(result) {
        clearEmptyStateIfNeeded(els.scenarioCards);
        const sign = result.scale_pct >= 0 ? '+' : '';
        const driver = result.driver_feature.replace(/_/g, ' ');
        const target = result.target_feature.replace(/_/g, ' ');
        const stName = result.station.replace(/_/g, ' ');
        const title = `What-If · ${target} · ${stName}`;
        const subtitle = `${sign}${result.scale_pct}% ${driver} for ${result.duration_months} month(s) · ${result.model}`;

        const cardId = `scenario-${Date.now()}`;
        const card = buildBaseCard(cardId, title, subtitle);
        const body = card.querySelector('.workspace-card-body');
        const plot = body.querySelector('.plot-container');

        const setupEl = document.createElement('div');
        setupEl.className = 'scenario-setup-grid';
        setupEl.innerHTML = `
            <div class="scenario-setup-item">
                <span class="scenario-setup-label">Station</span>
                <span class="scenario-setup-value">${escapeHtml(stName)}</span>
            </div>
            <div class="scenario-setup-item">
                <span class="scenario-setup-label">Target</span>
                <span class="scenario-setup-value">${escapeHtml(target)}</span>
            </div>
            <div class="scenario-setup-item">
                <span class="scenario-setup-label">Driver Adjustment</span>
                <span class="scenario-setup-value">${escapeHtml(sign + result.scale_pct + '% ' + driver)}</span>
            </div>
            <div class="scenario-setup-item">
                <span class="scenario-setup-label">Intervention Window</span>
                <span class="scenario-setup-value">Month ${result.start_offset + 1} to Month ${result.start_offset + result.duration_months}</span>
            </div>
            <div class="scenario-setup-item">
                <span class="scenario-setup-label">Response Model</span>
                <span class="scenario-setup-value">${result.sensitivity?.direct ? 'Direct scaling' : 'Lagged monthly anomaly response'}</span>
            </div>`;
        body.insertBefore(setupEl, plot);

        // Stats bar
        const stats = result.stats;
        const statsEl = document.createElement('div');
        statsEl.className = 'scenario-stats-bar';
        const fmt = v => (v >= 0 ? '+' : '') + v.toFixed(2);
        const fmtPct = v => (v >= 0 ? '+' : '') + v.toFixed(1) + '%';
        statsEl.innerHTML = `
            <div class="scenario-stat">
                <span class="scenario-stat-label">Mean delta</span>
                <span class="scenario-stat-value">${fmt(stats.mean_delta)} ${escapeHtml(result.unit)}</span>
            </div>
            <div class="scenario-stat">
                <span class="scenario-stat-label">Peak delta</span>
                <span class="scenario-stat-value">${fmt(stats.max_delta)} ${escapeHtml(result.unit)}</span>
            </div>
            <div class="scenario-stat">
                <span class="scenario-stat-label">Mean Δ%</span>
                <span class="scenario-stat-value ${stats.mean_delta_pct >= 0 ? 'positive' : 'negative'}">${fmtPct(stats.mean_delta_pct)}</span>
            </div>
            ${!result.sensitivity.direct ? `
            <div class="scenario-stat">
                <span class="scenario-stat-label">Elasticity</span>
                <span class="scenario-stat-value">${result.sensitivity.elasticity.toFixed(2)}</span>
            </div>
            <div class="scenario-stat">
                <span class="scenario-stat-label">Fit R</span>
                <span class="scenario-stat-value">${result.sensitivity.r_value.toFixed(2)}</span>
            </div>
            <div class="scenario-stat">
                <span class="scenario-stat-label">Dominant lag</span>
                <span class="scenario-stat-value">${result.sensitivity.dominant_lag} mo</span>
            </div>` : ''}
            <div class="scenario-stat scenario-stat--source">
                <span class="scenario-stat-label">Baseline</span>
                <span class="scenario-stat-value ${result.baseline_source === 'statistical_mean_fallback' ? 'scenario-source-fallback' : 'scenario-source-trained'}">${result.baseline_source === 'statistical_mean_fallback' ? 'Statistical mean' : 'Model forecast'}</span>
            </div>`;
        body.insertBefore(statsEl, plot);

        const plotLabel = document.createElement('div');
        plotLabel.className = 'chart-section-label';
        plotLabel.textContent = 'Forecast comparison';
        body.insertBefore(plotLabel, plot);

        const readingGuide = document.createElement('div');
        readingGuide.className = 'scenario-reading-guide';
        readingGuide.innerHTML = `
            <strong>How to read:</strong> blue dashed line = baseline forecast, orange line = scenario-adjusted forecast, shaded band = months where the driver shock is applied, lower bars = absolute target change from baseline. ${escapeHtml(result.model_note || '')}
        `;
        body.insertBefore(readingGuide, plot);

        // Show fallback warning banner if no trained CSV was used
        if (result.baseline_source === 'statistical_mean_fallback') {
            const fallbackBanner = document.createElement('div');
            fallbackBanner.className = 'scenario-fallback-banner';
            fallbackBanner.innerHTML = `⚠️ <strong>Statistical baseline</strong> — No trained model artifacts found for this station / feature combination. The scenario baseline uses the <strong>last-12-months mean</strong> instead of a trained ML forecast. Results are indicative only.`;
            body.insertBefore(fallbackBanner, plot);
        }

        els.scenarioCards.prepend(card);
        renderPlot(plot, result.figure);
        refreshPlotGrid(els.scenarioCards);

        // Analysis section (if included) — after the plot
        if (result.analysis) {
            const analysisEl = document.createElement('div');
            analysisEl.className = 'scenario-analysis-section';
            // Backend returns pre-rendered HTML (from markdown.markdown() or fallback HTML builder).
            // Inject directly — do NOT escapeHtml() first or the tags become visible raw text.
            const looksLikeHtml = /<\/?(p|ul|li|strong|em|h[1-6])\b/i.test(result.analysis);
            const content = looksLikeHtml
                ? result.analysis
                : escapeHtml(result.analysis).replace(/\n/g, '<br>').replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
            analysisEl.innerHTML = `
                <div class="analysis-header">
                    <h4>Interpretation</h4>
                </div>
                <div class="analysis-content">
                    ${content}
                </div>`;
            body.appendChild(analysisEl);
        }
    }

    // ── Quality Dashboard ─────────────────────────────────────────────────────

    function populateQualityControls() {
        if (!els.qualityStationSelect) return;
        const stations = (state.bootstrap?.stations || [])
            .filter(s => s.dataset === 'mekong')
            .sort((a, b) => a.name.localeCompare(b.name));
        els.qualityStationSelect.innerHTML = stations
            .map(s => `<option value="${escapeHtml(s.station)}">${escapeHtml(s.name.replace(/_/g, ' '))}</option>`)
            .join('');

        // Dataset picker for imputation view
        if (els.qualityDatasetPicker) {
            ['mekong', 'lamah'].forEach(ds => {
                const btn = document.createElement('button');
                btn.className = 'dataset-btn' + (ds === 'mekong' ? ' active' : '');
                btn.dataset.dataset = ds;
                btn.textContent = ds === 'mekong' ? 'Mekong' : 'LamaH-CE';
                btn.addEventListener('click', () => {
                    els.qualityDatasetPicker.querySelectorAll('.dataset-btn').forEach(b => b.classList.remove('active'));
                    btn.classList.add('active');
                    updateQualityImpFeatureOptions(ds);
                });
                els.qualityDatasetPicker.appendChild(btn);
            });
        }

        // Imputation feature select — seeded with Mekong features (active dataset)
        updateQualityImpFeatureOptions('mekong');

        updateQualityFeatureOptions();
        updateQualityControls('completeness');
    }

    function updateQualityImpFeatureOptions(dataset) {
        if (!els.qualityImpFeatureSelect) return;
        const datasetFeatures = state.bootstrap?.dataset_features || {};
        const features = (datasetFeatures[dataset] || state.bootstrap?.features || []).slice().sort();
        els.qualityImpFeatureSelect.innerHTML = '<option value="">All features</option>' +
            features.map(f => `<option value="${escapeHtml(f)}">${escapeHtml(f.replace(/_/g, ' '))}</option>`).join('');
    }

    function updateQualityFeatureOptions() {
        if (!els.qualityStationSelect || !els.qualityFeatureSelect) return;
        const station = els.qualityStationSelect.value;
        const stInfo = (state.bootstrap?.stations || []).find(s => s.station === station);
        const features = stInfo?.features || [];
        els.qualityFeatureSelect.innerHTML = features
            .map(f => `<option value="${escapeHtml(f)}">${escapeHtml(f.replace(/_/g, ' '))}</option>`)
            .join('');
    }

    function updateQualityControls(view) {
        const isStation = view === 'completeness' || view === 'gaps' || view === 'anomalies';
        els.qualityStationGroup?.classList.toggle('hidden', !isStation);
        els.qualityDatasetGroup?.classList.toggle('hidden', view !== 'imputation');
        els.qualityZGroup?.classList.toggle('hidden', view !== 'anomalies');
    }

    function qualityActiveView() {
        return els.qualityViewPicker?.querySelector('.mode-seg-btn.active')?.dataset.view || 'completeness';
    }

    async function runQualityAnalysis() {
        if (!els.runQualityBtn) return;
        const view = qualityActiveView();
        const includeAnalysis = document.getElementById('qualityAnalysisToggle')?.checked ?? false;
        showMessage(els.qualityMessage, 'Running…', '');
        els.runQualityBtn.disabled = true;

        let qualityResult = null;
        try {
            if (view === 'completeness') {
                const station = els.qualityStationSelect?.value;
                const feature = els.qualityFeatureSelect?.value;
                if (!station || !feature) throw new Error('Select a station and feature.');
                const res = await fetch(`/api/quality/completeness?station=${encodeURIComponent(station)}&feature=${encodeURIComponent(feature)}&include_analysis=${includeAnalysis}`);
                const data = await res.json();
                if (!data.ok) throw new Error(data.error);
                renderQualityCompleteness(data.result);
                qualityResult = data.result;
            } else if (view === 'imputation') {
                const dataset = els.qualityDatasetPicker?.querySelector('.dataset-btn.active')?.dataset.dataset || 'mekong';
                const feature = els.qualityImpFeatureSelect?.value || '';
                const params = new URLSearchParams({ dataset, include_analysis: includeAnalysis });
                if (feature) params.set('feature', feature);
                const res = await fetch(`/api/quality/imputation?${params}`);
                const data = await res.json();
                if (!data.ok) throw new Error(data.error);
                renderQualityImputation(data.result);
                qualityResult = data.result;
            } else if (view === 'gaps') {
                const station = els.qualityStationSelect?.value;
                const feature = els.qualityFeatureSelect?.value;
                if (!station || !feature) throw new Error('Select a station and feature.');
                const res = await fetch(`/api/quality/gaps?station=${encodeURIComponent(station)}&feature=${encodeURIComponent(feature)}&include_analysis=${includeAnalysis}`);
                const data = await res.json();
                if (!data.ok) throw new Error(data.error);
                renderQualityGaps(data.result);
                qualityResult = data.result;
            } else if (view === 'anomalies') {
                const station = els.qualityStationSelect?.value;
                const feature = els.qualityFeatureSelect?.value;
                const zThresh = els.qualityZSlider?.value || 3;
                if (!station || !feature) throw new Error('Select a station and feature.');
                const res = await fetch(`/api/quality/anomalies?station=${encodeURIComponent(station)}&feature=${encodeURIComponent(feature)}&z_thresh=${zThresh}&include_analysis=${includeAnalysis}`);
                const data = await res.json();
                if (!data.ok) throw new Error(data.error);
                renderQualityAnomalies(data.result);
                qualityResult = data.result;
            }
            activateDockTab('quality');
            showMessage(els.qualityMessage, 'Done.', 'success');
        } catch (err) {
            showMessage(els.qualityMessage, err.message || 'Analysis failed.', 'error');
        } finally {
            els.runQualityBtn.disabled = false;
        }
    }

    function qualitySection(title, note) {
        const section = document.createElement('div');
        section.className = 'quality-section';
        const h = document.createElement('h4');
        h.className = 'quality-section-title';
        h.textContent = title;
        section.appendChild(h);
        if (note) {
            const p = document.createElement('p');
            p.className = 'quality-note';
            p.textContent = note;
            section.appendChild(p);
        }
        return section;
    }

    function qualityStatChips(chips) {
        const bar = document.createElement('div');
        bar.className = 'quality-chips';
        chips.forEach(({ label, value, color }) => {
            const chip = document.createElement('div');
            chip.className = 'quality-chip';
            chip.innerHTML = `<span class="quality-chip-label">${escapeHtml(label)}</span>
                              <span class="quality-chip-value" style="${color ? `color:${color}` : ''}">${escapeHtml(String(value))}</span>`;
            bar.appendChild(chip);
        });
        return bar;
    }

    function renderQualityCompleteness(r) {
        if (!els.qualityWorkspace) return;
        const section = qualitySection(
            `Completeness · ${r.feature.replace(/_/g,' ')} · ${r.station.replace(/_/g,' ')}`,
            null
        );
        section.appendChild(qualityStatChips([
            { label: 'Overall', value: r.overall_pct + '%', color: r.overall_pct >= 80 ? '#34d399' : r.overall_pct >= 50 ? '#f59e0b' : '#ef4444' },
            { label: 'Missing months', value: r.missing_months },
            { label: 'Low (<50%)', value: r.low_months },
            { label: 'Total months', value: r.total_months },
        ]));
        const plotDiv = document.createElement('div');
        plotDiv.className = 'quality-plot';
        section.appendChild(plotDiv);
        appendAnalysisSection(section, r.analysis, { title: 'Interpretation' });
        els.qualityWorkspace.prepend(section);
        renderPlot(plotDiv, r.figure);
    }

    function renderQualityImputation(r) {
        if (!els.qualityWorkspace) return;
        const featLabel = r.feature ? r.feature.replace(/_/g,' ') : 'all features';
        const section = qualitySection(
            `Imputation Summary · ${r.dataset} · ${featLabel}`,
            null
        );
        section.appendChild(qualityStatChips([
            { label: 'Overall imp.', value: r.overall_imp_pct + '%', color: r.overall_imp_pct >= 20 ? '#ef4444' : r.overall_imp_pct >= 5 ? '#f59e0b' : '#34d399' },
            { label: 'Total obs.', value: r.total_observations.toLocaleString() },
            { label: 'Total imputed', value: r.total_imputed.toLocaleString() },
            { label: 'Stations w/ imp.', value: r.stations_with_imputation },
            { label: 'High imp. (≥20%)', value: r.high_imputation_stations },
        ]));
        const plotDiv = document.createElement('div');
        plotDiv.className = 'quality-plot';
        section.appendChild(plotDiv);

        // Full table (scrollable, capped at 50)
        const tableWrap = document.createElement('div');
        tableWrap.className = 'quality-table-wrap';
        const rows = r.rows.slice(0, 50);
        tableWrap.innerHTML = `<table class="quality-table">
            <thead><tr><th>Station</th><th>Feature</th><th>Observations</th><th>Imputed</th><th>Rate</th></tr></thead>
            <tbody>${rows.map(row => `
                <tr>
                    <td>${escapeHtml(row.name.replace(/_/g, ' '))}</td>
                    <td>${escapeHtml(row.feature.replace(/_/g,' '))}</td>
                    <td>${row.observations.toLocaleString()}</td>
                    <td>${row.imputed.toLocaleString()}</td>
                    <td><div class="quality-imp-bar-cell">
                        <div class="quality-imp-bar-bg"><div class="quality-imp-bar-fill" style="width:${Math.min(row.imp_pct,100)}%;background:${row.imp_pct>=20?'#ef4444':row.imp_pct>=5?'#f59e0b':'#38bdf8'}"></div></div>
                        <span>${row.imp_pct.toFixed(1)}%</span>
                    </div></td>
                </tr>`).join('')}
            </tbody></table>`;
        section.appendChild(tableWrap);
        appendAnalysisSection(section, r.analysis, { title: 'Interpretation' });
        els.qualityWorkspace.prepend(section);
        renderPlot(plotDiv, r.figure);
    }

    function renderQualityGaps(r) {
        if (!els.qualityWorkspace) return;
        const section = qualitySection(
            `Gap Detection · ${r.feature.replace(/_/g,' ')} · ${r.station.replace(/_/g,' ')}`,
            'Shaded regions on the chart mark the 5 largest gaps. Red = major (≥30 days), amber = moderate (7–29 days), gray = minor.'
        );
        section.appendChild(qualityStatChips([
            { label: 'Missing', value: r.missing_pct + '%', color: r.missing_pct >= 20 ? '#ef4444' : r.missing_pct >= 5 ? '#f59e0b' : '#34d399' },
            { label: 'Total gaps', value: r.gap_count },
            { label: 'Major', value: r.major, color: r.major > 0 ? '#ef4444' : undefined },
            { label: 'Moderate', value: r.moderate, color: r.moderate > 0 ? '#f59e0b' : undefined },
            { label: 'Minor', value: r.minor },
        ]));
        const plotDiv = document.createElement('div');
        plotDiv.className = 'quality-plot';
        section.appendChild(plotDiv);

        if (r.gaps.length) {
            const tableWrap = document.createElement('div');
            tableWrap.className = 'quality-table-wrap';
            tableWrap.innerHTML = `<table class="quality-table">
                <thead><tr><th>Start</th><th>End</th><th>Length</th><th>Severity</th></tr></thead>
                <tbody>${r.gaps.map(g => `
                    <tr>
                        <td>${g.start}</td><td>${g.end}</td>
                        <td>${g.length} ${g.unit}</td>
                        <td><span class="quality-severity-badge quality-sev-${g.severity}">${g.severity}</span></td>
                    </tr>`).join('')}
                </tbody></table>`;
            section.appendChild(tableWrap);
        }
        appendAnalysisSection(section, r.analysis, { title: 'Interpretation' });
        els.qualityWorkspace.prepend(section);
        renderPlot(plotDiv, r.figure);
    }

    function renderQualityAnomalies(r) {
        if (!els.qualityWorkspace) return;
        const section = qualitySection(
            `Anomaly Candidates · ${r.feature.replace(/_/g,' ')} · ${r.station.replace(/_/g,' ')}`,
            `Z-score threshold: ${r.z_thresh}  ·  Median: ${r.mean}  ·  MAD: ${r.std}  ·  ${r.unflagged} unflagged of ${r.total} candidates.`
        );

        if (!r.candidates.length) {
            const p = document.createElement('p');
            p.className = 'quality-note';
            p.textContent = 'No anomaly candidates found at this threshold.';
            section.appendChild(p);
            appendAnalysisSection(section, r.analysis, { title: 'Interpretation' });
            els.qualityWorkspace.prepend(section);
            return;
        }

        section.appendChild(qualityStatChips([
            { label: 'Total candidates', value: r.total },
            { label: 'Unflagged', value: r.unflagged },
            { label: 'Flagged', value: r.total - r.unflagged },
        ]));

        const PAGE_SIZE = 50;
        let currentPage = 0;

        function buildRows(candidates) {
            return candidates.map(c => {
                const direction = c.above_mean ? '▲ above median' : '▼ below median';
                const impBadge = c.is_imputed ? '<span class="quality-imp-badge">Imputed</span>' : '';
                return `<tr data-station="${escapeHtml(r.station)}" data-feature="${escapeHtml(r.feature)}" data-date="${escapeHtml(c.date)}">
                    <td>${c.date}${impBadge}</td>
                    <td class="quality-z-val">${c.value} ${escapeHtml(r.unit)}</td>
                    <td><span class="quality-zscore">${c.z_score}</span></td>
                    <td class="quality-dir">${direction}</td>
                    <td class="quality-flag-cell">
                        <button class="quality-flag-btn ${c.flag === 'real' ? 'active' : ''}" data-flag="real" title="Real event">Real</button>
                        <button class="quality-flag-btn ${c.flag === 'sensor_error' ? 'active' : ''}" data-flag="sensor_error" title="Sensor error">Error</button>
                        <button class="quality-flag-btn ${c.flag === 'uncertain' ? 'active' : ''}" data-flag="uncertain" title="Uncertain">?</button>
                        ${c.flag !== 'none' ? `<button class="quality-flag-btn quality-flag-clear" data-flag="none" title="Clear flag">✕</button>` : ''}
                    </td>
                </tr>`;
            }).join('');
        }

        const tableWrap = document.createElement('div');
        tableWrap.className = 'quality-table-wrap';

        const pageSlice = r.candidates.slice(0, PAGE_SIZE);
        tableWrap.innerHTML = `<table class="quality-table quality-anomaly-table">
            <thead><tr><th>Date</th><th>Value</th><th>|Z|</th><th>Direction</th><th>Flag</th></tr></thead>
            <tbody>${buildRows(pageSlice)}</tbody></table>`;

        // Pagination controls
        if (r.candidates.length > PAGE_SIZE) {
            const pageInfo = document.createElement('div');
            pageInfo.className = 'quality-page-info';
            const updatePageInfo = () => {
                const start = currentPage * PAGE_SIZE + 1;
                const end = Math.min((currentPage + 1) * PAGE_SIZE, r.candidates.length);
                pageInfo.textContent = `Showing ${start}–${end} of ${r.candidates.length} candidates`;
            };
            updatePageInfo();

            const pageBtns = document.createElement('div');
            pageBtns.className = 'quality-page-btns';
            const prevBtn = document.createElement('button');
            prevBtn.className = 'quality-page-btn';
            prevBtn.textContent = '← Prev';
            const nextBtn = document.createElement('button');
            nextBtn.className = 'quality-page-btn';
            nextBtn.textContent = 'Next →';

            const renderPage = () => {
                const slice = r.candidates.slice(currentPage * PAGE_SIZE, (currentPage + 1) * PAGE_SIZE);
                tableWrap.querySelector('tbody').innerHTML = buildRows(slice);
                prevBtn.disabled = currentPage === 0;
                nextBtn.disabled = (currentPage + 1) * PAGE_SIZE >= r.candidates.length;
                updatePageInfo();
            };

            prevBtn.addEventListener('click', () => { currentPage--; renderPage(); });
            nextBtn.addEventListener('click', () => { currentPage++; renderPage(); });
            prevBtn.disabled = true;
            pageBtns.appendChild(prevBtn);
            pageBtns.appendChild(nextBtn);
            tableWrap.appendChild(pageInfo);
            tableWrap.appendChild(pageBtns);
        }

        // Delegate flag button clicks
        tableWrap.addEventListener('click', async (e) => {
            const btn = e.target.closest('.quality-flag-btn');
            if (!btn) return;
            const row = btn.closest('tr');
            const station = row.dataset.station;
            const feature = row.dataset.feature;
            const date = row.dataset.date;
            const flag = btn.dataset.flag;
            try {
                const res = await fetch('/api/quality/flag', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ station, feature, date, flag }),
                });
                const data = await res.json();
                if (!data.ok) throw new Error(data.error);
                // Update button states in this row
                row.querySelectorAll('.quality-flag-btn:not(.quality-flag-clear)').forEach(b => b.classList.toggle('active', b.dataset.flag === flag && flag !== 'none'));
                // Show/hide clear button
                let clearBtn = row.querySelector('.quality-flag-clear');
                if (flag === 'none') {
                    clearBtn?.remove();
                } else if (!clearBtn) {
                    clearBtn = document.createElement('button');
                    clearBtn.className = 'quality-flag-btn quality-flag-clear';
                    clearBtn.dataset.flag = 'none';
                    clearBtn.title = 'Clear flag';
                    clearBtn.textContent = '✕';
                    row.querySelector('.quality-flag-cell').appendChild(clearBtn);
                }
            } catch (err) {
                console.error('Flag save failed:', err.message);
            }
        });

        section.appendChild(tableWrap);
        appendAnalysisSection(section, r.analysis, { title: 'Interpretation' });
        els.qualityWorkspace.prepend(section);
    }

    // ── Extreme Event Analysis ────────────────────────────────────────────────

    function populateExtremeControls() {
        if (!els.extremeStationSelect) return;
        const mekong = (state.bootstrap?.stations || [])
            .filter(s => s.dataset === 'mekong')
            .sort((a, b) => a.name.localeCompare(b.name));
        const lamah = (state.bootstrap?.stations || [])
            .filter(s => s.dataset === 'lamah')
            .sort((a, b) => a.name.localeCompare(b.name));

        const mkOpts = mekong.map(s =>
            `<option value="${escapeHtml(s.station)}">${escapeHtml(s.name.replace(/_/g, ' '))}</option>`
        ).join('');
        const laOpts = lamah.map(s =>
            `<option value="${escapeHtml(s.station)}">${escapeHtml(s.name.replace(/_/g, ' '))}</option>`
        ).join('');

        els.extremeStationSelect.innerHTML =
            `<optgroup label="Mekong">${mkOpts}</optgroup>` +
            `<optgroup label="LamaH-CE">${laOpts}</optgroup>`;

        updateExtremeFeatureOptions();
    }

    function updateExtremeFeatureOptions() {
        if (!els.extremeStationSelect || !els.extremeFeatureSelect) return;
        const station = els.extremeStationSelect.value;
        const stInfo = (state.bootstrap?.stations || []).find(s => s.station === station);
        const features = stInfo?.features || [];
        els.extremeFeatureSelect.innerHTML = features
            .map(f => `<option value="${escapeHtml(f)}"${f === 'Discharge' ? ' selected' : ''}>${escapeHtml(f.replace(/_/g, ' '))}</option>`)
            .join('');
    }

    async function runExtremeAnalysis() {
        if (!els.runExtremeBtn) return;
        const station = els.extremeStationSelect?.value;
        const feature = els.extremeFeatureSelect?.value;
        const distribution = els.extremeDistSelect?.value || 'gev';

        if (!station || !feature) {
            showMessage(els.extremeMessage, 'Select a station and feature.', 'error');
            return;
        }
        const includeAnalysis = document.getElementById('extremeAnalysisToggle')?.checked ?? false;
        showMessage(els.extremeMessage, includeAnalysis ? 'Fitting distributions + generating analysis…' : 'Fitting distributions…', '');
        els.runExtremeBtn.disabled = true;

        try {
            const params = new URLSearchParams({ station, feature, distribution, include_analysis: includeAnalysis });
            const res = await fetch(`/api/extreme?${params}`);
            const data = await res.json();
            if (!data.ok) throw new Error(data.error);
            appendExtremeCard(data.result);
            activateDockTab('extreme');
            showMessage(els.extremeMessage, `Done — ${data.result.n_years} yr of data fitted.`, 'success');
        } catch (err) {
            showMessage(els.extremeMessage, err.message || 'Analysis failed.', 'error');
        } finally {
            els.runExtremeBtn.disabled = false;
        }
    }

    function appendExtremeCard(result) {
        clearEmptyStateIfNeeded(els.extremeCards);
        const featureLabel = result.feature.replace(/_/g, ' ');
        const title = `Extreme Events · ${featureLabel} · ${result.station.replace(/_/g, ' ')}`;
        const subtitle = `${result.n_years} years (${result.year_range[0]}–${result.year_range[1]}) · ${result.unit || 'no unit'}`;

        const cardId = `extreme-${Date.now()}`;
        const card = buildBaseCard(cardId, title, subtitle);
        const body = card.querySelector('.workspace-card-body');
        const plot = body.querySelector('.plot-container');

        // Distribution parameter badges
        const paramsEl = document.createElement('div');
        paramsEl.className = 'extreme-params';
        if (result.gev_params) {
            const xi = result.gev_params.shape;
            const distType = xi > 0.05 ? 'Fréchet tail' : (xi < -0.05 ? 'Weibull tail' : 'Gumbel-approx');
            paramsEl.innerHTML += `<span class="extreme-param-badge">GEV ξ=${xi.toFixed(3)} (${escapeHtml(distType)})</span>`;
        }
        if (result.gumbel_params) {
            paramsEl.innerHTML += `<span class="extreme-param-badge">Gumbel μ=${result.gumbel_params.loc.toFixed(2)}, σ=${result.gumbel_params.scale.toFixed(2)}</span>`;
        }

        // Return levels table
        const hasGev = result.return_levels.some(r => 'gev' in r);
        const hasGumbel = result.return_levels.some(r => 'gumbel' in r);
        const hasCi = hasGev && result.ci_lower && result.ci_upper;

        let headerCols = '<th>Return Period</th>';
        if (hasGev) headerCols += `<th>GEV (${escapeHtml(result.unit)})</th>`;
        if (hasCi) headerCols += '<th>95% CI</th>';
        if (hasGumbel) headerCols += `<th>Gumbel (${escapeHtml(result.unit)})</th>`;

        const tableRows = result.return_levels.map((row, i) => {
            let cells = `<td><strong>${row.return_period} yr</strong></td>`;
            if (hasGev) cells += `<td>${row.gev !== undefined ? row.gev.toFixed(2) : '—'}</td>`;
            if (hasCi) {
                const lo = result.ci_lower[i] !== undefined ? result.ci_lower[i].toFixed(2) : '—';
                const hi = result.ci_upper[i] !== undefined ? result.ci_upper[i].toFixed(2) : '—';
                cells += `<td class="extreme-ci">[${lo}, ${hi}]</td>`;
            }
            if (hasGumbel) cells += `<td>${row.gumbel !== undefined ? row.gumbel.toFixed(2) : '—'}</td>`;
            return `<tr>${cells}</tr>`;
        }).join('');

        const tableSection = document.createElement('div');
        tableSection.className = 'extreme-table-section';
        tableSection.innerHTML = `
            <div class="extreme-table-wrap">
                <table class="extreme-table">
                    <thead><tr>${headerCols}</tr></thead>
                    <tbody>${tableRows}</tbody>
                </table>
            </div>`;

        body.insertBefore(paramsEl, plot);
        body.insertBefore(tableSection, plot);

        els.extremeCards.prepend(card);
        renderPlot(plot, result.figure);
        refreshPlotGrid(els.extremeCards);
        appendAnalysisSection(body, result.analysis, { title: 'Analysis Report' });
    }

    // ── Flood & Drought Risk Map ──────────────────────────────────────────────

    function populateRiskControls() {
        if (!els.riskFeatureSelect) return;
        updateRiskFeatureOptions();
    }

    function updateRiskFeatureOptions() {
        if (!els.riskFeatureSelect || !els.riskDatasetSelect) return;
        const dataset = els.riskDatasetSelect.value || 'mekong';
        const stations = (state.bootstrap?.stations || []).filter(s => s.dataset === dataset);
        const features = [...new Set(stations.flatMap(s => s.features))].sort();
        els.riskFeatureSelect.innerHTML = features
            .map(f => `<option value="${escapeHtml(f)}"${f === 'Discharge' ? ' selected' : ''}>${escapeHtml(f.replace(/_/g, ' '))}</option>`)
            .join('');
    }

    async function runRiskMap() {
        if (!els.runRiskBtn) return;
        const dataset = els.riskDatasetSelect?.value || 'mekong';
        const feature = els.riskFeatureSelect?.value;
        const lookback = Number(els.riskLookbackSlider?.value || 30);

        if (!feature) {
            showMessage(els.riskMessage, 'Select a feature.', 'error');
            return;
        }
        const includeAnalysis = document.getElementById('riskAnalysisToggle')?.checked ?? false;
        showMessage(els.riskMessage, 'Computing risk levels…', '');
        els.runRiskBtn.disabled = true;

        try {
            const params = new URLSearchParams({ dataset, feature, lookback, include_analysis: includeAnalysis });
            const res = await fetch(`/api/risk?${params}`);
            const data = await res.json();
            if (!data.ok) throw new Error(data.error);
            renderRiskMap(data.result);
            if (data.result.analysis && els.riskWorkspace) {
                appendAnalysisSection(els.riskWorkspace, data.result.analysis, { title: 'Analysis Report' });
            }
            activateDockTab('risk');
            showMessage(els.riskMessage, `${data.result.n_stations} stations classified.`, 'success');
        } catch (err) {
            showMessage(els.riskMessage, err.message || 'Risk map failed.', 'error');
        } finally {
            els.runRiskBtn.disabled = false;
        }
    }

    function renderRiskMap(r) {
        if (!els.riskWorkspace) return;
        els.riskWorkspace.innerHTML = '';

        const riskConfig = [
            { key: 'flood',          color: '#f87171', label: 'Flood Risk' },
            { key: 'flood_watch',    color: '#60a5fa', label: 'Flood Watch' },
            { key: 'normal',         color: '#34d399', label: 'Normal' },
            { key: 'drought',        color: '#fb923c', label: 'Drought' },
            { key: 'severe_drought', color: '#b91c1c', label: 'Severe Drought' },
        ];

        // Summary chips
        const chipsEl = document.createElement('div');
        chipsEl.className = 'risk-summary-chips';
        riskConfig.forEach(({ key, color, label }) => {
            const count = r.summary[key] || 0;
            const chip = document.createElement('div');
            chip.className = 'risk-chip';
            chip.style.setProperty('--risk-color', color);
            chip.innerHTML = `
                <span class="risk-chip-dot"></span>
                <span class="risk-chip-label">${escapeHtml(label)}</span>
                <span class="risk-chip-count">${count}</span>`;
            chipsEl.appendChild(chip);
        });

        // Map section
        const mapSection = document.createElement('div');
        mapSection.className = 'risk-section';

        const meta = document.createElement('p');
        meta.className = 'risk-note';
        meta.textContent = `${r.n_stations} stations · ${r.feature.replace(/_/g, ' ')} · lookback ${r.lookback} data points`;

        const plotDiv = document.createElement('div');
        plotDiv.className = 'risk-map';

        mapSection.appendChild(meta);
        mapSection.appendChild(plotDiv);

        els.riskWorkspace.appendChild(chipsEl);
        els.riskWorkspace.appendChild(mapSection);

        // Render Plotly map
        renderPlot(plotDiv, r.figure);
    }

    // ════════════════════════════════════════════════════════════════════════
    // CLIMATE CHANGE IMPACT PROJECTOR
    // ════════════════════════════════════════════════════════════════════════

    function updateSelectOptions(dataset, stationSelect, featureSelect) {
        if (!stationSelect) return;
        const stations = (state.bootstrap?.stations || []).filter(s => s.dataset === dataset);
        stationSelect.innerHTML = stations
            .map(s => `<option value="${escapeHtml(s.station)}">${escapeHtml((s.name || s.station).replace(/_/g, ' '))}</option>`)
            .join('');
        if (stationSelect.options.length > 0) {
            updateFeatureSelectForStation(dataset, stationSelect.value, featureSelect);
        }
    }

    function updateFeatureSelectForStation(dataset, station, featureSelect) {
        if (!featureSelect || !station) return;
        const stInfo = (state.bootstrap?.stations || []).find(s => s.station === station && s.dataset === dataset);
        let features = stInfo?.features || [];

        // Determine analysis context from select ID to filter mathematically valid features
        const sid = featureSelect.id;
        let analysisType = null;
        if (sid === 'climateFeatureSelect') analysisType = 'climate';
        else if (sid === 'extremeFeatureSelect') analysisType = 'extreme';
        else if (sid === 'riskFeatureSelect') analysisType = 'risk';
        else if (sid === 'networkContribGroup' || sid === 'networkContribStation') analysisType = 'network';
        
        if (analysisType && state.bootstrap?.feature_registry) {
            const reg = state.bootstrap.feature_registry;
            const allowed = reg.capabilities[analysisType];
            if (allowed && allowed.length > 0) {
                features = features.filter(f => {
                    const fType = reg.feature_type_map[f] || 'unknown';
                    return allowed.includes(fType);
                });
            }
        }

        if (features.length === 0) {
            featureSelect.innerHTML = '<option value="" disabled selected>No valid features for this tool</option>';
            return;
        }

        featureSelect.innerHTML = features
            .map(f => `<option value="${escapeHtml(f)}"${f === 'Discharge' ? ' selected' : ''}>${escapeHtml(f.replace(/_/g, ' '))}</option>`)
            .join('');
    }

    function initClimateControls() {
        if (!els.climateDatasetSelect) return;
        updateSelectOptions(els.climateDatasetSelect.value, els.climateStationSelect, els.climateFeatureSelect);
        setEmptyState(els.climateCards, 'No projections yet. Select a station and feature, then click Generate projection.');
    }

    async function runClimateProjection() {
        if (!els.runClimateBtn) return;
        const dataset = els.climateDatasetSelect?.value || 'mekong';
        const station = els.climateStationSelect?.value;
        const feature = els.climateFeatureSelect?.value;
        const projection_years = Number(els.climateYearsSlider?.value || 30);
        if (!station || !feature) {
            showMessage(els.climateMessage, 'Select a station and feature.', 'error');
            return;
        }
        const includeAnalysis = document.getElementById('climateAnalysisToggle')?.checked ?? false;
        showMessage(els.climateMessage, 'Projecting climate impacts… this may take a moment.', '');
        els.runClimateBtn.disabled = true;
        try {
            const res = await fetch('/api/climate-project', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ dataset, station, feature, projection_years, include_analysis: includeAnalysis }),
            });
            const data = await res.json();
            if (!data.ok) throw new Error(data.error);
            appendClimateCard(data.result);
            activateDockTab('climate');
            const s = data.result.stats;
            showMessage(els.climateMessage, `Projection added · trend ${s.historical_trend_per_decade > 0 ? '+' : ''}${s.historical_trend_per_decade}/decade (R²=${s.r_squared})`, 'success');
        } catch (err) {
            showMessage(els.climateMessage, err.message || 'Projection failed.', 'error');
        } finally {
            els.runClimateBtn.disabled = false;
        }
    }

    function appendClimateCard(result) {
        clearEmptyStateIfNeeded(els.climateCards);
        const cardId = `climate-${++state.cardCounters.climate}`;
        const card = buildBaseCard(cardId, result.title, result.subtitle);
        const body = card.querySelector('.workspace-card-body');
        els.climateCards.prepend(card);
        renderPlot(card.querySelector('.plot-container'), result.figure);
        appendAnalysisSection(body, result.analysis, { title: 'Analysis Report' });
    }

    // ════════════════════════════════════════════════════════════════════════
    // CHANGE POINT DETECTION
    // ════════════════════════════════════════════════════════════════════════

    function initChangepointControls() {
        if (!els.cpDatasetSelect) return;
        updateSelectOptions(els.cpDatasetSelect.value, els.cpStationSelect, els.cpFeatureSelect);
        setEmptyState(els.changepointCards, 'No analyses yet. Select a station and feature, then click Detect change points.');
    }

    async function runChangePointDetection() {
        if (!els.runCpBtn) return;
        const dataset = els.cpDatasetSelect?.value || 'mekong';
        const station = els.cpStationSelect?.value;
        const feature = els.cpFeatureSelect?.value;
        const n_breaks = Number(els.cpBreaksSlider?.value || 3);
        const method = els.cpMethodPicker?.querySelector('.mode-seg-btn.active')?.dataset.method || 'pelt';
        if (!station || !feature) {
            showMessage(els.cpMessage, 'Select a station and feature.', 'error');
            return;
        }
        const includeAnalysis = document.getElementById('cpAnalysisToggle')?.checked ?? false;
        showMessage(els.cpMessage, 'Detecting structural breaks… this may take a moment.', '');
        els.runCpBtn.disabled = true;
        try {
            const res = await fetch('/api/changepoints', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ dataset, station, feature, n_breaks, method, include_analysis: includeAnalysis }),
            });
            const data = await res.json();
            if (!data.ok) throw new Error(data.error);
            appendChangepointCard(data.result);
            activateDockTab('changepoint');
            const cps = data.result.stats.change_point_dates?.join(', ') || 'none';
            showMessage(els.cpMessage, `${data.result.stats.n_breaks_detected} break(s) detected: ${cps}`, 'success');
        } catch (err) {
            showMessage(els.cpMessage, err.message || 'Detection failed.', 'error');
        } finally {
            els.runCpBtn.disabled = false;
        }
    }

    function appendChangepointCard(result) {
        clearEmptyStateIfNeeded(els.changepointCards);
        const cardId = `cp-${++state.cardCounters.changepoint}`;
        const card = buildBaseCard(cardId, result.title, result.subtitle);
        const body = card.querySelector('.workspace-card-body');
        els.changepointCards.prepend(card);
        renderPlot(card.querySelector('.plot-container'), result.figure);
        refreshPlotGrid(els.changepointCards);
        appendAnalysisSection(body, result.analysis, { title: 'Analysis Report' });
    }

    // ════════════════════════════════════════════════════════════════════════
    // ANIMATED TIME-SERIES MAP
    // ════════════════════════════════════════════════════════════════════════

    function updateAnimateFeatureOptions() {
        if (!els.animateFeatureSelect || !els.animateDatasetSelect) return;
        const dataset = els.animateDatasetSelect.value || 'mekong';
        const stations = (state.bootstrap?.stations || []).filter(s => s.dataset === dataset);
        const features = [...new Set(stations.flatMap(s => s.features))].sort();
        els.animateFeatureSelect.innerHTML = features
            .map(f => `<option value="${escapeHtml(f)}"${f === 'Discharge' ? ' selected' : ''}>${escapeHtml(f.replace(/_/g, ' '))}</option>`)
            .join('');
    }

    function initAnimateControls() {
        if (!els.animateDatasetSelect) return;
        updateAnimateFeatureOptions();
        setEmptyState(els.animateCards, 'No animations yet. Select a dataset and feature, then click Build animation.');
    }

    async function runAnimatedMap() {
        if (!els.runAnimateBtn) return;
        const dataset = els.animateDatasetSelect?.value || 'mekong';
        const feature = els.animateFeatureSelect?.value;
        if (!feature) {
            showMessage(els.animateMessage, 'Select a feature.', 'error');
            return;
        }
        showMessage(els.animateMessage, 'Building animated map… this may take a moment.', '');
        els.runAnimateBtn.disabled = true;
        try {
            const speed = Number(document.getElementById('animateSpeedSlider')?.value ?? 2);
            const params = new URLSearchParams({ dataset, feature, speed });
            const res = await fetch(`/api/animate-map?${params}`);
            const data = await res.json();
            if (!data.ok) throw new Error(data.error);
            const frameDurations = [1000, 500, 250, 125, 62];
            appendAnimateCard(data.result, frameDurations[speed - 1]);
            activateDockTab('animate');
            showMessage(els.animateMessage, `Animation ready · ${data.result.stats.n_stations} stations · ${data.result.stats.n_years} years`, 'success');
        } catch (err) {
            showMessage(els.animateMessage, err.message || 'Animation failed.', 'error');
        } finally {
            els.runAnimateBtn.disabled = false;
        }
    }

    function appendAnimateCard(result, frameDuration = 500) {
        clearEmptyStateIfNeeded(els.animateCards);
        els.animateCards?.querySelectorAll('.workspace-card').forEach(card => card.remove());
        const cardId = `animate-${++state.cardCounters.animate}`;
        const card = buildBaseCard(cardId, result.title, result.subtitle);
        card.querySelector('.card-action-btn[title="Expand to fullscreen"]')?.remove();
        const plotContainer = card.querySelector('.plot-container');
        const originalHeight = Number(result?.figure?.layout?.height || 720);
        const lockAnimateHeight = () => {
            if (!plotContainer) return;
            plotContainer.style.height = `${originalHeight}px`;
            plotContainer.style.minHeight = `${originalHeight}px`;
            if (plotContainer.layout) {
                const updateLayout = { autosize: true, height: originalHeight };
                delete updateLayout.width;
                Plotly.relayout(plotContainer, updateLayout).catch(() => {});
            }
        };
        if (plotContainer) {
            plotContainer.style.height = `${originalHeight}px`;
            plotContainer.style.minHeight = `${originalHeight}px`;
        }
        els.animateCards.prepend(card);
        renderPlot(plotContainer, result.figure);
        refreshPlotGrid(els.animateCards);
        requestAnimationFrame(() => {
            lockAnimateHeight();
            setTimeout(lockAnimateHeight, 80);
            setTimeout(lockAnimateHeight, 220);
        });

        // Custom play/pause toggle button — placed below the plot
        const btn = document.createElement('button');
        btn.className = 'anim-play-btn';
        btn.innerHTML = '&#9654; Play';
        let playing = false;

        function stopAnim() {
            playing = false;
            btn.innerHTML = '&#9654; Play';
            Plotly.animate(plotContainer, [null], {
                frame: { duration: 0, redraw: false },
                mode: 'immediate',
                transition: { duration: 0 },
            });
            lockAnimateHeight();
            setTimeout(lockAnimateHeight, 80);
        }

        btn.addEventListener('click', () => {
            if (!plotContainer._fullLayout) return; // Plotly not ready yet
            if (!playing) {
                playing = true;
                btn.innerHTML = '&#9646;&#9646; Pause';
                lockAnimateHeight();
                Plotly.animate(plotContainer, null, {
                    frame: { duration: frameDuration, redraw: true },
                    fromcurrent: true,
                    transition: { duration: Math.min(frameDuration * 0.4, 200) },
                }).then(() => {
                    // Reached end naturally
                    if (playing) stopAnim();
                });
                setTimeout(lockAnimateHeight, 80);
                setTimeout(lockAnimateHeight, 220);
            } else {
                stopAnim();
            }
        });

        const btnRow = document.createElement('div');
        btnRow.className = 'anim-play-row';
        btnRow.appendChild(btn);
        card.querySelector('.workspace-card-body')?.appendChild(btnRow) || card.appendChild(btnRow);
    }

    // ════════════════════════════════════════════════════════════════════════
    // MULTI-MODEL FORECAST COMPARISON
    // ════════════════════════════════════════════════════════════════════════

    function initModelCompareControls() {
        if (!els.mcDatasetSelect) return;
        updateModelCompareOptions();
        setEmptyState(els.mcCards, 'No comparisons yet. Select a station and feature, then click Compare models.');
    }

    async function updateModelCompareOptions() {
        if (!els.mcDatasetSelect || !els.mcStationSelect || !els.mcFeatureSelect) return;

        const dataset = els.mcDatasetSelect.value || 'mekong';
        updateSelectOptions(dataset, els.mcStationSelect, els.mcFeatureSelect);

        const feature = els.mcFeatureSelect.value;
        if (!feature) return;

        try {
            const params = new URLSearchParams({ dataset, feature });
            const res = await fetch(`/api/model-compare-stations?${params}`);
            const data = await res.json();
            if (!data.ok) throw new Error(data.error || 'Failed to load supported stations.');

            const supported = new Set(data.stations || []);
            const stations = (state.bootstrap?.stations || [])
                .filter(s => s.dataset === dataset && supported.has(s.station));

            els.mcStationSelect.innerHTML = stations
                .map(s => `<option value="${escapeHtml(s.station)}">${escapeHtml((s.name || s.station).replace(/_/g, ' '))}</option>`)
                .join('');

            if (els.mcStationSelect.options.length === 0) {
                els.mcStationSelect.innerHTML = '<option value="" disabled selected>No supported stations for these models</option>';
                return;
            }

            updateFeatureSelectForStation(dataset, els.mcStationSelect.value, els.mcFeatureSelect);
            if (els.mcFeatureSelect.querySelector(`option[value="${CSS.escape(feature)}"]`)) {
                els.mcFeatureSelect.value = feature;
            }
        } catch (_err) {
            // Fall back to the generic station list if capability lookup fails.
        }
    }

    async function runModelComparison() {
        if (!els.runMcBtn) return;
        const dataset = els.mcDatasetSelect?.value || 'mekong';
        const station = els.mcStationSelect?.value;
        const feature = els.mcFeatureSelect?.value;
        const horizon = Number(els.mcHorizonSlider?.value || 12);
        if (!station || !feature) {
            showMessage(els.mcMessage, 'Select a station and feature.', 'error');
            return;
        }
        const includeAnalysis = document.getElementById('mcAnalysisToggle')?.checked ?? false;
        showMessage(els.mcMessage, 'Fitting models… this may take a moment.', '');
        els.runMcBtn.disabled = true;
        try {
            const res = await fetch('/api/model-compare', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ dataset, station, feature, horizon, include_analysis: includeAnalysis }),
            });
            const data = await res.json();
            if (!data.ok) throw new Error(data.error);
            appendMcCard(data.result);
            activateDockTab('modelcompare');
            showMessage(els.mcMessage, `Done · best model: ${data.result.stats.best_model_by_rmse}`, 'success');
        } catch (err) {
            showMessage(els.mcMessage, err.message || 'Comparison failed.', 'error');
        } finally {
            els.runMcBtn.disabled = false;
        }
    }

    function appendMcCard(result) {
        clearEmptyStateIfNeeded(els.mcCards);
        const cardId = `mc-${++state.cardCounters.modelcompare}`;
        const card = buildBaseCard(cardId, result.title, result.subtitle);
        const body = card.querySelector('.workspace-card-body');

        // Metrics table
        if (result.stats?.models?.length) {
            const hasAnyData = result.stats.models.some(m => m.source_note !== 'no_data');
            const hasPartialData = hasAnyData && result.stats.models.some(m => m.source_note === 'no_data');
            const table = document.createElement('table');
            table.className = 'metrics-table';
            table.innerHTML = `
                <thead><tr><th>Model</th><th>RMSE</th><th>MAPE</th></tr></thead>
                <tbody>${result.stats.models.map((m) => {
                    const isBest = m.Model === result.stats.best_model_by_rmse;
                    const isNoData = m.source_note === 'no_data';
                    const rowClass = isBest ? ' class="best-row"' : isNoData ? ' class="no-data-row"' : '';
                    return `<tr${rowClass}>
                        <td>${escapeHtml(m.Model)}${isBest ? ' ✓' : ''}</td>
                        <td>${escapeHtml(m.RMSE)}</td>
                        <td>${escapeHtml(m.MAPE)}</td>
                    </tr>`;
                }).join('')}</tbody>`;
            body.appendChild(table);
            if (hasPartialData) {
                const note = document.createElement('div');
                note.className = 'mc-partial-note';
                note.innerHTML = `ℹ️ Some models have no trained artifacts for this station / feature and are excluded from comparison.`;
                body.appendChild(note);
            }
        }
        // Zoom plot container
        if (result.figure_zoom) {
            const zoomDiv = document.createElement('div');
            zoomDiv.className = 'plot-container';
            zoomDiv.style.minHeight = '340px';
            body.appendChild(zoomDiv);
        }

        els.mcCards.prepend(card);
        renderPlot(card.querySelector('.plot-container'), result.figure);
        if (result.figure_zoom) {
            const plots = card.querySelectorAll('.plot-container');
            renderPlot(plots[plots.length - 1], result.figure_zoom);
        }
        refreshPlotGrid(els.mcCards);
        appendAnalysisSection(body, result.analysis, { title: 'Analysis Report' });
    }

    // ════════════════════════════════════════════════════════════════════════
    // STL DECOMPOSITION
    // ════════════════════════════════════════════════════════════════════════

    function initDecomposeControls() {
        if (!els.decompDatasetSelect) return;
        updateSelectOptions(els.decompDatasetSelect.value, els.decompStationSelect, els.decompFeatureSelect);
        setEmptyState(els.decompCards, 'No decompositions yet. Select a station and feature, then click Decompose series.');
    }

    async function runDecomposition() {
        if (!els.runDecompBtn) return;
        const dataset = els.decompDatasetSelect?.value || 'mekong';
        const station = els.decompStationSelect?.value;
        const feature = els.decompFeatureSelect?.value;
        if (!station || !feature) {
            showMessage(els.decompMessage, 'Select a station and feature.', 'error');
            return;
        }
        const includeAnalysis = document.getElementById('decompAnalysisToggle')?.checked ?? false;
        showMessage(els.decompMessage, 'Decomposing series…', '');
        els.runDecompBtn.disabled = true;
        try {
            const res = await fetch('/api/decompose', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ dataset, station, feature, include_analysis: includeAnalysis }),
            });
            const data = await res.json();
            if (!data.ok) throw new Error(data.error);
            appendDecompCard(data.result);
            activateDockTab('decompose');
            const s = data.result.stats;
            showMessage(els.decompMessage,
                `Done · trend strength ${s.strength_trend} · seasonal strength ${s.strength_seasonal} · peak: ${s.seasonal_peak_month}`,
                'success');
        } catch (err) {
            showMessage(els.decompMessage, err.message || 'Decomposition failed.', 'error');
        } finally {
            els.runDecompBtn.disabled = false;
        }
    }

    function appendDecompCard(result) {
        clearEmptyStateIfNeeded(els.decompCards);
        const cardId = `decomp-${++state.cardCounters.decompose}`;
        const card = buildBaseCard(cardId, result.title, result.subtitle);
        const body = card.querySelector('.workspace-card-body');
        const plotContainer = card.querySelector('.plot-container');
        if (plotContainer) plotContainer.style.minHeight = '580px';
        els.decompCards.prepend(card);
        renderPlot(plotContainer, result.figure);
        refreshPlotGrid(els.decompCards);
        appendAnalysisSection(body, result.analysis, { title: 'Analysis Report' });
    }

    // ════════════════════════════════════════════════════════════════════════
    // WAVELET ANALYSIS
    // ════════════════════════════════════════════════════════════════════════

    function initWaveletControls() {
        if (!els.waveletDatasetSelect) return;
        updateSelectOptions(els.waveletDatasetSelect.value, els.waveletStationSelect, els.waveletFeatureSelect);
        setEmptyState(els.waveletCards, 'No analyses yet. Select a station and feature, then click Run wavelet analysis.');
    }

    async function runWaveletAnalysis() {
        if (!els.runWaveletBtn) return;
        const dataset = els.waveletDatasetSelect?.value || 'mekong';
        const station = els.waveletStationSelect?.value;
        const feature = els.waveletFeatureSelect?.value;
        if (!station || !feature) {
            showMessage(els.waveletMessage, 'Select a station and feature.', 'error');
            return;
        }
        const includeAnalysis = document.getElementById('waveletAnalysisToggle')?.checked ?? false;
        showMessage(els.waveletMessage, 'Running wavelet transform… this may take a moment.', '');
        els.runWaveletBtn.disabled = true;
        try {
            const res = await fetch('/api/wavelet', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ dataset, station, feature, include_analysis: includeAnalysis }),
            });
            const data = await res.json();
            if (!data.ok) throw new Error(data.error);
            appendWaveletCard(data.result);
            activateDockTab('wavelet');
            const dp = data.result.stats.dominant_periods_months?.join(', ') || '—';
            showMessage(els.waveletMessage, `Done · dominant periods: ${dp} months`, 'success');
        } catch (err) {
            showMessage(els.waveletMessage, err.message || 'Wavelet analysis failed.', 'error');
        } finally {
            els.runWaveletBtn.disabled = false;
        }
    }

    function appendWaveletCard(result) {
        clearEmptyStateIfNeeded(els.waveletCards);
        const cardId = `wavelet-${++state.cardCounters.wavelet}`;
        const card = buildBaseCard(cardId, result.title, result.subtitle);
        const body = card.querySelector('.workspace-card-body');
        const plotContainer = card.querySelector('.plot-container');
        if (plotContainer) plotContainer.style.minHeight = '460px';

        const guide = document.createElement('div');
        guide.className = 'scenario-reading-guide';
        const dominant = result.stats?.dominant_periods_months?.length
            ? result.stats.dominant_periods_months.join(', ') + ' months'
            : 'not clearly identified';
        guide.innerHTML = `
            <strong>How to read:</strong> the top heatmap shows <strong>when</strong> repeating cycles were strongest, and the lower summary shows which cycle lengths are strongest <strong>overall</strong>. Brighter colors mean stronger repeating behavior. Bands near <strong>12 months</strong> suggest annual seasonality; longer periods suggest multi-year variability. Dominant periods here: ${escapeHtml(dominant)}.
        `;

        els.waveletCards.prepend(card);
        body.insertBefore(guide, plotContainer);
        renderPlot(plotContainer, result.figure);
        refreshPlotGrid(els.waveletCards);
        appendAnalysisSection(body, result.analysis, { title: 'Analysis Report' });
    }

    // ── Shared AI analysis renderer ───────────────────────────────────────────
    function appendAnalysisSection(body, analysisText, options = {}) {
        if (!analysisText || !body) return;
        const el = document.createElement('div');
        el.className = 'scenario-analysis-section';
        const looksLikeHtml = /<\/?(p|ul|li|h1|h2|h3|h4|strong|em|br)\b/i.test(analysisText);
        const formatted = looksLikeHtml
            ? analysisText
            : escapeHtml(analysisText)
                .replace(/\n/g, '<br>')
                .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
        el.innerHTML = `
            <div class="analysis-header">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="M12 16v-4M12 8h.01"/></svg>
                ${escapeHtml(options.title || 'AI Analysis')}
            </div>
            <div class="analysis-content">${formatted}</div>`;
        body.appendChild(el);
    }

})();
