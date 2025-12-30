const { useState, useEffect, useRef } = React;

// --- Components ---

const Header = () => (
    <header className="p-6 border-b border-surface bg-darker/50 backdrop-blur-md sticky top-0 z-50">
        <div className="container mx-auto flex items-center justify-between">
            <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-primary to-secondary flex items-center justify-center shadow-lg shadow-primary/20">
                    <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19.428 15.428a2 2 0 00-1.022-.547l-2.384-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" /></svg>
                </div>
                <h1 className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-emerald-400">
                    Pyrolysis<span className="text-slate-400 font-light">Sim</span>
                </h1>
            </div>
            <div className="text-sm text-slate-400">
                Advanced Transients Platform v1.0
            </div>
        </div>
    </header>
);

const LineChart = ({ title, data, labels, datasets }) => {
    const canvasRef = useRef(null);
    const chartRef = useRef(null);

    useEffect(() => {
        if (!canvasRef.current) return;

        const ctx = canvasRef.current.getContext('2d');

        if (chartRef.current) {
            chartRef.current.destroy();
        }

        chartRef.current = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: datasets.map(ds => ({
                    ...ds,
                    borderWidth: 2,
                    tension: 0.4,
                    pointRadius: 0,
                    pointHoverRadius: 4
                }))
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: {
                    mode: 'index',
                    intersect: false,
                },
                plugins: {
                    legend: {
                        position: 'top',
                        labels: { color: '#94a3b8' }
                    },
                    title: {
                        display: false,
                    }
                },
                scales: {
                    x: {
                        grid: { color: '#334155' },
                        ticks: { color: '#94a3b8' }
                    },
                    y: {
                        grid: { color: '#334155' },
                        ticks: { color: '#94a3b8' }
                    }
                }
            }
        });

        return () => {
            if (chartRef.current) chartRef.current.destroy();
        };
    }, [data, labels, datasets]);

    return (
        <div className="glass-panel p-6 rounded-2xl h-80 flex flex-col">
            <h3 className="text-lg font-medium text-slate-200 mb-4">{title}</h3>
            <div className="flex-1 min-h-0">
                <canvas ref={canvasRef}></canvas>
            </div>
        </div>
    );
};

const MetricCard = ({ label, value, unit, color }) => (
    <div className="glass-panel p-6 rounded-2xl relative overflow-hidden group">
        <div className={`absolute top-0 right-0 p-4 opacity-10 group-hover:opacity-20 transition-opacity`}>
            <div className={`w-16 h-16 rounded-full bg-${color}-500 blur-xl`}></div>
        </div>
        <div className="relative z-10">
            <p className="text-sm text-slate-400 font-medium mb-1">{label}</p>
            <div className="flex items-baseline gap-1">
                <span className="text-3xl font-bold text-white">{value}</span>
                <span className="text-sm text-slate-500">{unit}</span>
            </div>
        </div>
    </div>
);

const App = () => {
    const [loading, setLoading] = useState(false);
    const [params, setParams] = useState({
        tau_max: 10.0,
        initial_temp: 0.5,
        initial_moisture: 0.4
    });
    const [results, setResults] = useState(null);
    const [error, setError] = useState(null);

    const runSimulation = async () => {
        setLoading(true);
        setError(null);
        try {
            const response = await fetch('/simulate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(params)
            });

            if (!response.ok) throw new Error('Simulation failed');

            const data = await response.json();
            setResults(data);
        } catch (err) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    // Run initial simulation
    useEffect(() => {
        runSimulation();
    }, []);

    // Prepare chart data
    const time = results?.pyrolysis.time || [];

    const compositionDatasets = results ? [
        { label: 'Biomass', data: results.pyrolysis.biomass, borderColor: '#3b82f6', backgroundColor: '#3b82f6' },
        { label: 'Bio-oil', data: results.pyrolysis.biooil, borderColor: '#10b981', backgroundColor: '#10b981' },
        { label: 'Gas', data: results.pyrolysis.gas, borderColor: '#ef4444', backgroundColor: '#ef4444' },
        { label: 'Char', data: results.pyrolysis.char, borderColor: '#64748b', backgroundColor: '#64748b' },
    ] : [];

    const temperatureDatasets = results ? [
        { label: 'Temperature', data: results.pyrolysis.temperature, borderColor: '#f59e0b', backgroundColor: '#f59e0b', yAxisID: 'y' },
        { label: 'Moisture', data: results.pyrolysis.moisture, borderColor: '#06b6d4', backgroundColor: '#06b6d4', yAxisID: 'y1' }
    ] : [];

    const reactorDatasets = results ? [
        { label: 'Conversion', data: results.reactor.conversion, borderColor: '#8b5cf6', backgroundColor: '#8b5cf6' }
    ] : [];

    return (
        <div className="min-h-screen pb-20">
            <Header />

            <main className="container mx-auto px-6 py-8">

                {/* Controls Section */}
                <section className="mb-10">
                    <div className="glass-panel p-6 rounded-2xl">
                        <div className="flex flex-wrap items-end gap-6">
                            <div className="space-y-2">
                                <label className="text-sm font-medium text-slate-400">Max Time (τ)</label>
                                <input
                                    type="number"
                                    value={params.tau_max}
                                    onChange={e => setParams({ ...params, tau_max: parseFloat(e.target.value) })}
                                    className="block w-32 bg-slate-900 border border-slate-700 rounded-lg px-4 py-2 text-white focus:ring-2 focus:ring-primary focus:border-transparent outline-none transition-all"
                                />
                            </div>
                            <div className="space-y-2">
                                <label className="text-sm font-medium text-slate-400">Initial Temp (T*)</label>
                                <input
                                    type="number"
                                    step="0.1"
                                    value={params.initial_temp}
                                    onChange={e => setParams({ ...params, initial_temp: parseFloat(e.target.value) })}
                                    className="block w-32 bg-slate-900 border border-slate-700 rounded-lg px-4 py-2 text-white focus:ring-2 focus:ring-primary focus:border-transparent outline-none transition-all"
                                />
                            </div>
                            <button
                                onClick={runSimulation}
                                disabled={loading}
                                className={`px-8 py-2 rounded-lg font-medium text-white shadow-lg shadow-primary/25 transition-all transform hover:scale-105 active:scale-95 ${loading ? 'bg-slate-700 cursor-not-allowed' : 'bg-gradient-to-r from-primary to-indigo-600 hover:to-indigo-500'}`}
                            >
                                {loading ? 'Simulating...' : 'Run Simulation'}
                            </button>
                        </div>
                    </div>
                </section>

                {error && (
                    <div className="mb-8 p-4 bg-red-500/10 border border-red-500/20 rounded-xl text-red-400">
                        {error}
                    </div>
                )}

                {results && (
                    <div className="space-y-8 animate-fade-in">

                        {/* Metrics Grid */}
                        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                            <MetricCard
                                label="Bio-oil Yield"
                                value={results.pyrolysis.final_yields['Bio-oil'].toFixed(3)}
                                unit=""
                                color="emerald"
                            />
                            <MetricCard
                                label="Char Yield"
                                value={results.pyrolysis.final_yields['Char'].toFixed(3)}
                                unit=""
                                color="slate"
                            />
                            <MetricCard
                                label="Residual Biomass"
                                value={results.pyrolysis.final_yields['Biomasa residual'].toFixed(3)}
                                unit=""
                                color="blue"
                            />
                            <MetricCard
                                label="Final Conversion"
                                value={((1 - results.pyrolysis.final_yields['Biomasa residual']) * 100).toFixed(1)}
                                unit="%"
                                color="purple"
                            />
                        </div>

                        {/* Charts Grid */}
                        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                            <LineChart
                                title="Species Evolution"
                                labels={time}
                                datasets={compositionDatasets}
                            />
                            <LineChart
                                title="Thermal Dynamics"
                                labels={time}
                                datasets={temperatureDatasets}
                            />
                            <div className="lg:col-span-2">
                                <LineChart
                                    title="Reactor Conversion"
                                    labels={results.reactor.time}
                                    datasets={reactorDatasets}
                                />
                            </div>
                        </div>

                    </div>
                )}
            </main>
        </div>
    );
};

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(<App />);
