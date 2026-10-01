// Optional boot hook if you prefer not to edit CLIManager.cs.

using UnityEngine;

namespace AiCar
{
    [DefaultExecutionOrder(-1000)]
    public class MapBoot : MonoBehaviour
    {
        void Awake()
        {
            MapConfig.ApplyCommandLine();
        }
    }
}
