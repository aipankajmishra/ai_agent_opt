import numpy as np
from typing import Dict, Tuple

class ClaimRiskModel:
    """
    Simulates an XGBoost/Random Forest model that predicts claim risk.
    In production, this would be a trained model loaded from disk.
    """
    
    def __init__(self):
        self.feature_weights = {
            'amount_vs_avg': 0.35,
            'diagnosis_risk': 0.25,
            'provider_history': 0.20,
            'member_history': 0.15,
            'procedure_match': 0.05
        }
    
    def predict(self, claim_data: Dict) -> Tuple[float, Dict]:
        """
        Returns (risk_score, feature_importance)
        risk_score: 0.0 (low risk) to 1.0 (high risk)
        """
        
        billed = claim_data['billed_amount']
        approved = claim_data['approved_amount']
        member_avg = claim_data['member_avg_cost']
        diagnosis_risk = claim_data['diagnosis_risk_score']
        provider_score = claim_data['provider_reliability_score']
        
        features = {}
        
        amount_ratio = billed / max(approved, 1)
        features['amount_vs_avg'] = min(amount_ratio - 1.0, 1.0)
        
        features['diagnosis_risk'] = diagnosis_risk
        
        features['provider_history'] = 1.0 - provider_score
        
        cost_deviation = abs(billed - member_avg) / max(member_avg, 1)
        features['member_history'] = min(cost_deviation, 1.0)
        
        procedure_match = 1.0 if claim_data.get('procedure_code_valid', True) else 0.5
        features['procedure_match'] = 1.0 - procedure_match
        
        risk_score = sum(
            features[k] * self.feature_weights[k] 
            for k in features
        )
        
        risk_score = max(0.0, min(1.0, risk_score))
        
        feature_importance = {
            k: features[k] * self.feature_weights[k] 
            for k in features
        }
        
        return risk_score, feature_importance
    
    def get_decision(self, risk_score: float) -> str:
        """
        Auto-decision based on risk score
        """
        if risk_score < 0.3:
            return "AUTO_APPROVE"
        elif risk_score > 0.7:
            return "AUTO_DENY"
        else:
            return "NEEDS_REVIEW"

def prepare_claim_features(claim: Dict, member_history: Dict) -> Dict:
    """
    Extracts features from raw claim data for model input
    """
    
    diagnosis_risk_map = {
        "Angina Pectoris": 0.4,
        "Routine Health Checkup": 0.1,
        "Coronary Artery Disease": 0.8,
        "COPD Exacerbation": 0.5,
        "Type 2 Diabetes Management": 0.3
    }
    
    provider_reliability_map = {
        "City General Hospital": 0.9,
        "Wellness Clinic": 0.95,
        "Heart & Vascular Center": 0.85,
        "Respiratory Care Associates": 0.90,
        "Downtown Pharmacy": 0.98
    }
    
    return {
        'billed_amount': claim['billed_amount'],
        'approved_amount': claim['approved_amount'],
        'member_avg_cost': member_history['avg_monthly_cost'],
        'diagnosis_risk_score': diagnosis_risk_map.get(claim['diagnosis'], 0.5),
        'provider_reliability_score': provider_reliability_map.get(claim['provider'], 0.8),
        'procedure_code_valid': True
    }
