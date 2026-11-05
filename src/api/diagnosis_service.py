from threading import Lock


class UserNotFoundError(RuntimeError):
    pass


class UserSuspendedError(RuntimeError):
    pass


class FaultCatalogError(RuntimeError):
    pass


class DiagnosisService:
    def __init__(self, predictor, firebase_service):
        self.predictor = predictor
        self.firebase = firebase_service
        self._prediction_lock = Lock()

    @staticmethod
    def _public_fault(fault_id, data):
        return {
            'faultId': fault_id,
            'name': data.get('name'),
            'description': data.get('description'),
            'recommendation': data.get('recommendation'),
            'urgency': data.get('urgency'),
            'category': data.get('category'),
        }

    def diagnose(self, uid, symptom_answers, notes):
        try:
            self.firebase.assert_application_user_active(uid)
        except LookupError as error:
            raise UserNotFoundError(str(error)) from error
        except PermissionError as error:
            raise UserSuspendedError(str(error)) from error

        diagnosis_ref = self.firebase.create_processing_diagnosis(
            uid=uid,
            symptom_answers=symptom_answers,
            notes=notes,
        )

        try:
            with self._prediction_lock:
                result = self.predictor.predict(symptom_answers, top_k=3)

            fault_ids = [item['faultId'] for item in result['predictions']]
            try:
                faults = self.firebase.get_faults(fault_ids)
            except LookupError as error:
                raise FaultCatalogError(str(error)) from error

            self.firebase.complete_diagnosis(diagnosis_ref, result)

            enriched_predictions = []
            for item in result['predictions']:
                fault_id = item['faultId']
                fault = faults[fault_id]
                enriched_predictions.append({
                    'faultId': fault_id,
                    'confidence': item['confidence'],
                    'name': fault.get('name'),
                    'urgency': fault.get('urgency'),
                })

            predicted_fault_id = result['predictedFaultId']
            return {
                'diagnosisId': diagnosis_ref.id,
                'status': 'completed',
                'predictedFaultId': predicted_fault_id,
                'confidence': result['confidence'],
                'fault': self._public_fault(predicted_fault_id, faults[predicted_fault_id]),
                'predictions': enriched_predictions,
            }
        except Exception:
            try:
                self.firebase.fail_diagnosis(diagnosis_ref)
            except Exception:
                pass
            raise
