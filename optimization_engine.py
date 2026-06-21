import pulp
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate the great circle distance in kilometers between two points on the earth."""
    # Convert latitude and longitude to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371 # Radius of earth in kilometers
    return c * r

class MILPResourceOptimizer:
    """
    Mixed Integer Linear Programming (MILP) Resource Optimizer using PuLP.
    Minimizes response time while maximizing resource coverage for critical incidents.
    """
    def __init__(self, total_barricades=100):
        self.total_barricades = total_barricades

    def optimize(self, incidents, units):
        """
        incidents: List of dicts [{'id': 'inc_1', 'lat': 12.9, 'lon': 77.5, 'severity': 'CRITICAL', 'impact_score': 15}]
        units: List of dicts [{'id': 'unit_1', 'lat': 12.91, 'lon': 77.52}]
        """
        # Create the problem
        prob = pulp.LpProblem("Gridlock_Resource_Allocation", pulp.LpMaximize)

        # Variables
        x_vars = {} # x[i, j] = 1 if unit i assigned to incident j
        b_vars = {} # b[j] = number of barricades assigned to incident j

        # Sets & Parameters
        incident_ids = [inc['id'] for inc in incidents]
        unit_ids = [u['id'] for u in units]

        req_officers = {}
        req_barricades = {}
        priority = {}

        for inc in incidents:
            j = inc['id']
            sev = inc.get('severity', 'MINOR')
            impact = inc.get('impact_score', 1)
            
            if sev == 'CRITICAL':
                req_o, req_b, prio = 5, 20, 1000 + impact * 10
            elif sev == 'MODERATE':
                req_o, req_b, prio = 2, 10, 100 + impact * 10
            else:
                req_o, req_b, prio = 1, 0, 10 + impact * 10
                
            req_officers[j] = req_o
            req_barricades[j] = req_b
            priority[j] = prio
            # Determine required unit type based on event_cause
            cause = str(inc.get('event_cause', '')).lower()
            if 'riot' in cause or 'protest' in cause or 'rally' in cause:
                req_type = 'civil'
            elif 'crash' in cause or 'breakdown' in cause or 'accident' in cause or 'overturn' in cause:
                req_type = 'traffic'
            else:
                req_type = 'any'
            inc['_req_type'] = req_type
            
            b_vars[j] = pulp.LpVariable(f"b_{j}", lowBound=0, upBound=req_b, cat=pulp.LpInteger)

        for u in units:
            i = u['id']
            for inc in incidents:
                j = inc['id']
                x_vars[(i, j)] = pulp.LpVariable(f"x_{i}_{j}", cat=pulp.LpBinary)

        # Objective Function
        # Maximize coverage weighted by priority, subtract a small penalty for distance
        objective_terms = []
        
        # Coverage term
        for j in incident_ids:
            # Officer coverage proportion contribution
            if req_officers[j] > 0:
                officer_sum = pulp.lpSum([x_vars[(i, j)] for i in unit_ids])
                objective_terms.append( (priority[j] / req_officers[j]) * officer_sum )
                
            # Barricade coverage proportion contribution
            if req_barricades[j] > 0:
                objective_terms.append( (priority[j] / req_barricades[j]) * b_vars[j] )

        # Distance penalty term (scaled down so it doesn't overpower priority)
        distance_weight = 0.5 
        for u in units:
            i = u['id']
            for inc in incidents:
                j = inc['id']
                # Default lat/lon if missing
                u_lat, u_lon = u.get('lat', 12.9716), u.get('lon', 77.5946)
                inc_lat, inc_lon = inc.get('lat', 12.9716), inc.get('lon', 77.5946)
                
                dist = haversine_distance(u_lat, u_lon, inc_lat, inc_lon)
                objective_terms.append( -distance_weight * dist * x_vars[(i, j)] )

        prob += pulp.lpSum(objective_terms)

        # Constraints
        # 1. A unit can be assigned to at most 1 incident
        for i in unit_ids:
            prob += pulp.lpSum([x_vars[(i, j)] for j in incident_ids]) <= 1, f"Unit_{i}_max_1_assignment"

        # 2. An incident cannot receive more officers than required
        for j in incident_ids:
            prob += pulp.lpSum([x_vars[(i, j)] for i in unit_ids]) <= req_officers[j], f"Inc_{j}_max_officers"

        # 3. Total barricades across all incidents cannot exceed available
        prob += pulp.lpSum([b_vars[j] for j in incident_ids]) <= self.total_barricades, "Total_barricades_limit"

        # 4. Enforce unit_type matching (civil vs traffic)
        for u in units:
            i = u['id']
            u_type = str(u.get('unit_type', 'any')).lower()
            for inc in incidents:
                j = inc['id']
                r_type = inc['_req_type']
                if r_type != 'any' and u_type != 'any' and r_type != u_type:
                    prob += x_vars[(i, j)] == 0, f"TypeMismatch_{i}_{j}"

        # Solve
        prob.solve(pulp.PULP_CBC_CMD(msg=0))

        # Parse Results
        assignments = {}
        for j in incident_ids:
            assignments[j] = {
                'assigned_officers': [],
                'assigned_barricades': int(b_vars[j].varValue) if b_vars[j].varValue else 0,
                'req_officers': req_officers[j],
                'req_barricades': req_barricades[j],
                'status': pulp.LpStatus[prob.status]
            }

        for u in units:
            i = u['id']
            for j in incident_ids:
                if x_vars[(i, j)].varValue and x_vars[(i, j)].varValue > 0.5:
                    assignments[j]['assigned_officers'].append(i)

        return assignments

if __name__ == "__main__":
    # Test script
    optimizer = MILPResourceOptimizer(total_barricades=30)
    test_incidents = [
        {'id': 'inc1', 'lat': 12.9716, 'lon': 77.5946, 'severity': 'CRITICAL', 'impact_score': 15},
        {'id': 'inc2', 'lat': 12.9100, 'lon': 77.6000, 'severity': 'MODERATE', 'impact_score': 5}
    ]
    test_units = [
        {'id': f'unit_{i}', 'lat': 12.97 + (i*0.001), 'lon': 77.59, 'unit_type': 'traffic' if i % 2 == 0 else 'civil'} for i in range(5)
    ]
    
    res = optimizer.optimize(test_incidents, test_units)
    print("Optimization Results:")
    for inc_id, alloc in res.items():
        print(f"Incident {inc_id}: {len(alloc['assigned_officers'])} officers, {alloc['assigned_barricades']} barricades")
