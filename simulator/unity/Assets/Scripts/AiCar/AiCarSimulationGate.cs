using System;
using System.Collections;
using System.Globalization;
using SocketIO;
using UnityEngine;

namespace AiCar
{
    /// <summary>
    /// Preserves the legacy PPO pause/resume path unless explicit action stepping
    /// is opted into with AICAR_ACTION_INTERVAL_SECONDS. In that mode it gates
    /// the existing automatic Unity fixed-step loop; it does not simulate physics manually.
    /// </summary>
    [DefaultExecutionOrder(32000)]
    public sealed class AiCarSimulationGate : MonoBehaviour
    {
        const float PauseTimeScale = 0.001f;
        const string IntervalVariable = "AICAR_ACTION_INTERVAL_SECONDS";
        const string IdleTargetFrameRateVariable = "AICAR_ACTION_IDLE_TARGET_FPS";
        const string PauseEvent = "AICAR_SIMULATION_PAUSE";
        const string ResumeEvent = "AICAR_SIMULATION_RESUME";
        const string PauseAckEvent = "AICAR_SIMULATION_PAUSE_ACK";
        const string ResumeAckEvent = "AICAR_SIMULATION_RESUME_ACK";

        static AiCarSimulationGate _instance;
        float _normalTimeScale;
        float _normalFixedDeltaTime;
        float _normalCaptureDeltaTime;
        int _normalTargetFrameRate;
        bool _paused;
        bool _actionStepMode;
        bool _stepping;
        bool _waitingReset;
        bool _startupComponentsSettled;
        int _resetFrames;
        int _ticksPerAction;
        int _ticksThisAction;
        int _activeStepId;
        int _nextStepId = 1;
        double _actionInterval;
        double _simulatedSeconds;
        string _configurationError;
        bool _debugSteps;
        int _idleTargetFrameRate = 120;
        SocketIOComponent _socket;
        Action<int, double, int, string> _completed;

        public static AiCarSimulationGate Instance { get { return _instance; } }
        public static bool ActionStepMode { get { return _instance != null && _instance._actionStepMode; } }
        public double ActionIntervalSeconds { get { return _actionInterval; } }
        public double SimulatedSeconds { get { return _simulatedSeconds; } }
        public string ConfigurationError { get { return _configurationError; } }

        public bool ValidateStepId(int stepId, out string error)
        {
            error = null;
            if (!_actionStepMode) error = "explicit action-step mode is disabled";
            else if (!string.IsNullOrEmpty(_configurationError)) error = _configurationError;
            else if (_stepping || _waitingReset || _completed != null) error = "another action is pending";
            else if (stepId != _nextStepId) error = "step ID mismatch; expected " + _nextStepId;
            return error == null;
        }

        void Awake()
        {
            _instance = this;
            _normalTimeScale = Time.timeScale;
            _normalFixedDeltaTime = Time.fixedDeltaTime;
            _normalCaptureDeltaTime = Time.captureDeltaTime;
            _normalTargetFrameRate = Application.targetFrameRate;
            _debugSteps = Environment.GetEnvironmentVariable("AICAR_ACTION_STEP_DEBUG") == "1";
            string value = Environment.GetEnvironmentVariable(IntervalVariable);
            if (!string.IsNullOrWhiteSpace(value))
            {
                ConfigureActionStepMode(value);
                ConfigureIdleTargetFrameRate();
            }
        }

        void ConfigureIdleTargetFrameRate()
        {
            string value = Environment.GetEnvironmentVariable(IdleTargetFrameRateVariable);
            int requested;
            if (string.IsNullOrWhiteSpace(value)) return;
            if (!int.TryParse(value, NumberStyles.Integer, CultureInfo.InvariantCulture, out requested) || requested < 1)
            {
                Debug.LogWarning("[AiCar.SimulationGate] invalid " + IdleTargetFrameRateVariable + "; keeping 120 FPS");
                return;
            }
            _idleTargetFrameRate = requested;
        }

        void ConfigureActionStepMode(string value)
        {
            _actionStepMode = true;
            SocketIOComponent.PumpInUpdate = true;
            double requested;
            if (!double.TryParse(value, NumberStyles.Float, CultureInfo.InvariantCulture, out requested) ||
                double.IsNaN(requested) || double.IsInfinity(requested) || requested <= 0)
            {
                _configurationError = "AICAR_ACTION_INTERVAL_SECONDS must be finite and positive";
                Time.timeScale = 0f;
                Debug.LogError("[AiCar.SimulationGate] " + _configurationError);
                return;
            }
            double ratio = requested / _normalFixedDeltaTime;
            int ticks = (int)Math.Round(ratio);
            if (ticks < 1 || Math.Abs(ratio - ticks) > 1e-5)
            {
                _configurationError = "action interval must be an integer multiple of Unity fixedDeltaTime";
                Time.timeScale = 0f;
                Debug.LogError("[AiCar.SimulationGate] " + _configurationError);
                _actionInterval = requested;
                return;
            }
            _ticksPerAction = ticks;
            _actionInterval = ticks * (double)_normalFixedDeltaTime;
            if (_actionInterval + _normalFixedDeltaTime * 0.5 > Time.maximumDeltaTime + 1e-9)
            {
                _configurationError = "action interval plus half-tick alignment margin exceeds maximumDeltaTime";
                Time.timeScale = 0f;
                Debug.LogError("[AiCar.SimulationGate] " + _configurationError);
                return;
            }
            Time.captureDeltaTime = (float)_actionInterval;
            Time.timeScale = 0f;
            _paused = true;
            // Bound Update polling while waiting for Python without sleeping a Unity thread.
            Application.targetFrameRate = _idleTargetFrameRate;
            Debug.Log("[AiCar.SimulationGate] explicit mode: " + ticks + " fixed ticks/action");
        }

        IEnumerator Start()
        {
            while (_socket == null)
            {
                foreach (var candidate in Resources.FindObjectsOfTypeAll<SocketIOComponent>())
                {
                    if (candidate == null || !candidate.isActiveAndEnabled || !candidate.gameObject.scene.IsValid()) continue;
                    _socket = candidate;
                    break;
                }
                if (_socket == null) yield return null;
            }
            _socket.On(PauseEvent, OnPause);
            _socket.On(ResumeEvent, OnResume);
        }

        public void BeginActionStep(int stepId, bool reset, Action<int, double, int, string> completed)
        {
            string validationError;
            if (!ValidateStepId(stepId, out validationError))
            { completed?.Invoke(stepId, _simulatedSeconds, 0, validationError); return; }

            _activeStepId = stepId;
            if (_debugSteps) Debug.Log("[AiCar.SimulationGate] received step " + stepId + " reset=" + reset);
            _completed = completed;
            _ticksThisAction = 0;
            _waitingReset = reset;
            _resetFrames = reset ? 2 : 0;
            _paused = true;
            Time.timeScale = 0f;
            if (!reset) StartTickBatch();
        }

        void StartTickBatch()
        {
            // Recompute from Unity's double precision clocks each batch. The
            // half-tick remainder avoids float rounding at an exact boundary.
            double target = Time.fixedTimeAsDouble + (_ticksPerAction + 0.5) * _normalFixedDeltaTime;
            double frameDelta = target - Time.timeAsDouble;
            if (frameDelta <= 0 || frameDelta > Time.maximumDeltaTime + 1e-9)
            { Fail("cannot align fixed-tick batch within maximumDeltaTime"); return; }
            Time.captureDeltaTime = (float)frameDelta;
            _waitingReset = false;
            _stepping = true;
            _paused = false;
            Application.targetFrameRate = -1;
            Time.timeScale = _normalTimeScale > 0 ? _normalTimeScale : 1f;
            if (_debugSteps) Debug.Log("[AiCar.SimulationGate] started step " + _activeStepId + " targetTicks=" + _ticksPerAction + " captureDelta=" + frameDelta.ToString("R", CultureInfo.InvariantCulture) + " fixed=" + Time.fixedTimeAsDouble.ToString("R", CultureInfo.InvariantCulture) + " time=" + Time.timeAsDouble.ToString("R", CultureInfo.InvariantCulture));
        }

        void FixedUpdate()
        {
            if (!_actionStepMode || !_stepping) return;
            _ticksThisAction++;
            if (_debugSteps && (_ticksThisAction == 1 || _ticksThisAction == _ticksPerAction || _ticksThisAction > _ticksPerAction))
                Debug.Log("[AiCar.SimulationGate] step " + _activeStepId + " fixedTick=" + _ticksThisAction);
        }

        void Update()
        {
            if (!_actionStepMode) return;
            // TrackLoader and ResetManager Start methods have run by the first
            // Update. Physics is still held here, so SpawnKeeper has not had a
            // FixedUpdate in which it could teleport a vehicle. Its startup
            // safeguard is no longer needed after the authoritative spawn and
            // reset references are established.
            if (!_startupComponentsSettled)
            {
                foreach (var keeper in Resources.FindObjectsOfTypeAll<SpawnKeeper>())
                {
                    if (keeper != null && keeper.gameObject.scene.IsValid()) keeper.enabled = false;
                }
                foreach (var behaviour in Resources.FindObjectsOfTypeAll<MonoBehaviour>())
                {
                    if (behaviour != null && behaviour.gameObject.scene.IsValid() &&
                        behaviour.GetType().Name == "TimeScale")
                        behaviour.enabled = false;
                }
                _startupComponentsSettled = true;
            }
            if (_waitingReset)
            {
                // Two full Update barriers let ResetManager consume ResetFlag,
                // then restore dynamic rigidbodies before the first physics step.
                if (--_resetFrames <= 0) StartTickBatch();
                return;
            }
            if (_stepping && _ticksThisAction >= _ticksPerAction)
            {
                Time.timeScale = 0f;
                _paused = true;
                _stepping = false;
                if (_ticksThisAction != _ticksPerAction)
                { Fail("Unity executed " + _ticksThisAction + " fixed ticks; expected " + _ticksPerAction); return; }
                _simulatedSeconds += _actionInterval;
                _nextStepId = _activeStepId + 1;
                Complete(_ticksThisAction, null);
            }
        }

        void LateUpdate()
        {
            if (!_actionStepMode) return;
            // Own Unity's global clock after scene/UI Update scripts. This also
            // preserves the serialized fixed timestep if a time-scale slider
            // exists in a different AutoDRIVE scene.
            Time.fixedDeltaTime = _normalFixedDeltaTime;
            if (_stepping)
            {
                Time.timeScale = _normalTimeScale > 0 ? _normalTimeScale : 1f;
                Application.targetFrameRate = -1;
            }
            else
            {
                Time.timeScale = 0f;
                Application.targetFrameRate = _idleTargetFrameRate;
            }
        }

        void Complete(int ticks, string error)
        {
            if (_debugSteps) Debug.Log("[AiCar.SimulationGate] complete step " + _activeStepId + " ticks=" + ticks + " sim=" + _simulatedSeconds.ToString("R", CultureInfo.InvariantCulture) + " error=" + (error ?? ""));
            Application.targetFrameRate = _idleTargetFrameRate;
            var cb = _completed;
            _completed = null;
            cb?.Invoke(_activeStepId, _simulatedSeconds, ticks, error);
        }

        void Fail(string error)
        {
            _configurationError = error;
            Application.targetFrameRate = _idleTargetFrameRate;
            Time.timeScale = 0f;
            _paused = true;
            _stepping = false;
            _waitingReset = false;
            Complete(0, error);
        }

        void OnPause(SocketIOEvent _)
        {
            if (_actionStepMode) { _socket.Emit(PauseAckEvent); return; }
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
            if (_actionStepMode) { _socket.Emit(ResumeAckEvent); return; }
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
            if (_socket != null) { _socket.Off(PauseEvent, OnPause); _socket.Off(ResumeEvent, OnResume); }
            if (_instance == this) _instance = null;
            SocketIOComponent.PumpInUpdate = false;
            Time.fixedDeltaTime = _normalFixedDeltaTime;
            Time.captureDeltaTime = _normalCaptureDeltaTime;
            Time.timeScale = _normalTimeScale;
            Application.targetFrameRate = _normalTargetFrameRate;
        }
    }
}
