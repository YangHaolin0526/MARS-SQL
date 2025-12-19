package main

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"log"
	"math/rand"
	"net/http"
	"net/http/httputil"
	"net/url"
	"os"
	"strconv"
)

// Configuration from environment variables
var (
	port           = getEnvWithDefault("PORT", "3000")
	replicaCount   = getEnvAsIntWithDefault("REPLICA_COUNT", 4)
	backendService = getEnvWithDefault("BACKEND_SERVICE", "openhands")
	backendPort    = getEnvWithDefault("BACKEND_PORT", "3000")
	namespace      = getEnvWithDefault("NAMESPACE", "default")
)

func main() {
	log.Printf("Starting API Gateway on port %s with %d backends", port, replicaCount)

	// Set up route for image_exists
	http.HandleFunc("/image_exists", handleImageExists)

	// Default handler for other routes
	http.HandleFunc("/", handleDefault)

	// Start the server
	log.Fatal(http.ListenAndServe(":"+port, nil))
}

// handleImageExists specifically routes requests based on the image parameter
func handleImageExists(w http.ResponseWriter, r *http.Request) {
	// Get the image parameter
	imageName := r.URL.Query().Get("image")
	if imageName == "" {
		http.Error(w, "Missing image parameter", http.StatusBadRequest)
		return
	}

	// Determine which node should handle this image
	nodeIndex := getNodeForImage(imageName)
	backendURL := getBackendURL(nodeIndex)

	log.Printf("Routing image_exists for %s to node %d (%s)", imageName, nodeIndex, backendURL)

	// Create a reverse proxy to the target backend
	proxyToBackend(backendURL, w, r)
}

// handleDefault handles any other endpoints
func handleDefault(w http.ResponseWriter, r *http.Request) {
	// For other endpoints, just use a random node
	path := r.URL.Path
	nodeIndex := rand.Intn(replicaCount)
	backendURL := getBackendURL(nodeIndex)
	log.Printf("Routing %s request to node %d (%s)", path, nodeIndex, backendURL)
	proxyToBackend(backendURL, w, r)
}

// proxyToBackend forwards a request to a specific backend
func proxyToBackend(backendURL string, w http.ResponseWriter, r *http.Request) {
	// Parse the backend URL
	target, err := url.Parse(backendURL)
	if err != nil {
		log.Printf("Error parsing backend URL: %v", err)
		http.Error(w, "Internal server error", http.StatusInternalServerError)
		return
	}

	// Create the reverse proxy
	proxy := httputil.NewSingleHostReverseProxy(target)

	// Update the request's Host header
	r.Host = target.Host

	// Forward the request
	proxy.ServeHTTP(w, r)
}

// getNodeForImage provides consistent hashing for image names
func getNodeForImage(imageName string) int {
	hash := sha256.Sum256([]byte(imageName))
	hashHex := hex.EncodeToString(hash[:])

	// Take the first 8 characters of the hash as a hex number
	hashPrefix := hashHex[:8]
	hashNum, _ := strconv.ParseInt(hashPrefix, 16, 64)

	// Modulo to get the node index
	return int(hashNum % int64(replicaCount))
}

// getBackendURL constructs the backend service URL for a given node index
func getBackendURL(nodeIndex int) string {
	return fmt.Sprintf("http://%s-%d.%s.%s.svc.cluster.local:%s",
		backendService, nodeIndex, backendService, namespace, backendPort)
}

// Helper function to get an environment variable with a default value
func getEnvWithDefault(key, defaultValue string) string {
	if value, exists := os.LookupEnv(key); exists {
		return value
	}
	return defaultValue
}

// Helper function to get an environment variable as an integer with a default
func getEnvAsIntWithDefault(key string, defaultValue int) int {
	if value, exists := os.LookupEnv(key); exists {
		if intValue, err := strconv.Atoi(value); err == nil {
			return intValue
		}
	}
	return defaultValue
}
