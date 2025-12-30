from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional
import numpy as np
import threading
import sys
import os

# Add current directory to path so we can import the model
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pyrolysis_model import CompleteTransientPyrolysis

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

app = FastAPI(title="Pyrolysis Simulation API")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development, allow all
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

class SimulationConfig(BaseModel):
    tau_max: float = 10.0
    initial_temp: float = 0.5
    initial_moisture: float = 0.4
    # We can add more parameters as needed

class SimulationResult(BaseModel):
    time: List[float]
    biomass: List[float]
    biooil: List[float]
    gas: List[float]
    char: List[float]
    temperature: List[float]
    moisture: List[float]
    final_yields: Dict[str, float]

class ReactorResult(BaseModel):
    time: List[float]
    conversion: List[float]
    temperature: List[float]
    # We can add more fields from reactor results

class FullResult(BaseModel):
    pyrolysis: SimulationResult
    reactor: ReactorResult

@app.get("/")
def read_root():
    return FileResponse('static/index.html')

@app.post("/simulate", response_model=FullResult)
def run_simulation_endpoint(config: SimulationConfig):
    try:
        # Initialize system
        system = CompleteTransientPyrolysis()
        
        # Apply custom initial conditions if possible
        # We need to access the inner pyrolysis system to set initial conditions
        # The current class structure hardcodes them in __init__, so we modify them after init
        system.pyrolysis_system.initial_conditions['temp_star'] = config.initial_temp
        system.pyrolysis_system.initial_conditions['moisture_star'] = config.initial_moisture
        
        # Run simulation
        results = system.run_complete_transient_simulation(tau_max=config.tau_max)
        
        # Process Pyrolysis Results
        pyro_data = results['pyrolysis']
        pyrolysis_res = SimulationResult(
            time=pyro_data['time_star'].tolist(),
            biomass=pyro_data['biomass'].tolist(),
            biooil=pyro_data['biooil'].tolist(),
            gas=pyro_data['gas'].tolist(),
            char=pyro_data['char'].tolist(),
            temperature=pyro_data['temperature'].tolist(),
            moisture=pyro_data['moisture'].tolist(),
            final_yields=pyro_data['final_yields']
        )
        
        # Process Reactor Results
        reactor_data = results['reactor']
        reactor_res = ReactorResult(
            time=reactor_data['time_star'].tolist(),
            conversion=reactor_data['conversion'].tolist(),
            temperature=reactor_data['temperature'].tolist()
        )
        
        return FullResult(pyrolysis=pyrolysis_res, reactor=reactor_res)

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

def run_single(tau):
    system = CompleteTransientPyrolysis()
    res = system.run_complete_transient_simulation(tau_max=tau)
    final_biooil = res['pyrolysis']['final_yields']['Bio-oil']
    final_biomass = res['pyrolysis']['final_yields']['Biomasa residual']
    return {
        "tau": tau,
        "final_biooil": final_biooil,
        "final_biomass": final_biomass
    }

@app.post("/sweep")
def run_sweep_endpoint(tau_values: List[float]):
    """
    Run parallel simulations for multiple tau values (Sensitivity Analysis)
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed
    
    results_summary = []
    
    # Limit workers
    with ProcessPoolExecutor(max_workers=min(len(tau_values), 4)) as executor:
        futures = [executor.submit(run_single, tau) for tau in tau_values]
        for future in as_completed(futures):
            results_summary.append(future.result())
            
    # Sort by tau
    results_summary.sort(key=lambda x: x['tau'])
    
    return results_summary

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
