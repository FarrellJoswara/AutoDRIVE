using System.Collections;
using SocketIO;
using UnityEngine;

namespace AiCar
{
    /// <summary>
    /// Keeps physics stationary while PPO updates its policy. The Socket.IO
    /// fixed-update pump must continue running so it can receive a resume event,
    /// so pause uses a tiny time scale with a proportionally adjusted fixed step.
    /// </summary>
    public sealed class AiCarSimulationGate : MonoBehaviour
    {
        const float PauseTimeScale = 0.001f;
        const string PauseEvent = "AICAR_SIMULATION_PAUSE";
        const string ResumeEvent = "AICAR_SIMULATION_RESUME";
        const string PauseAckEvent = "AICAR_SIMULATION_PAUSE_ACK";
        const string ResumeAckEvent = "AICAR_SIMULATION_RESUME_ACK";

        float _normalTimeScale;
        float _normalFixedDeltaTime;
        bool _paused;
        SocketIOComponent _socket;

        void Awake()
        {
            _normalTimeScale = Mathf.Max(Time.timeScale, 0.0001f);
            _normalFixedDeltaTime = Time.fixedDeltaTime;
        }

        IEnumerator Start()
        {
            while (_socket == null)
            {
                var sockets = Resources.FindObjectsOfTypeAll<SocketIOComponent>();
                foreach (var candidate in sockets)
                {
                    if (
                        candidate == null
                        || !candidate.isActiveAndEnabled
                        || !candidate.gameObject.scene.IsValid()
                    )
                        continue;
                    _socket = candidate;
                    break;
                }

                if (_socket == null)
                    yield return null;
            }

            _socket.On(PauseEvent, OnPause);
            _socket.On(ResumeEvent, OnResume);
            Debug.Log("[AiCar.SimulationGate] pause/resume control registered");
        }

        void OnPause(SocketIOEvent _)
        {
            if (!_paused)
            {
                Time.fixedDeltaTime = _normalFixedDeltaTime * PauseTimeScale / _normalTimeScale;
                Time.timeScale = PauseTimeScale;
                _paused = true;
            }
            _socket.Emit(PauseAckEvent);
        }

        void OnResume(SocketIOEvent _)
        {
            if (_paused)
            {
                Time.fixedDeltaTime = _normalFixedDeltaTime;
                Time.timeScale = _normalTimeScale;
                _paused = false;
            }
            _socket.Emit(ResumeAckEvent);
        }

        void OnDestroy()
        {
            if (_socket == null)
                return;
            _socket.Off(PauseEvent, OnPause);
            _socket.Off(ResumeEvent, OnResume);
            if (_paused)
            {
                Time.fixedDeltaTime = _normalFixedDeltaTime;
                Time.timeScale = _normalTimeScale;
            }
        }
    }
}
